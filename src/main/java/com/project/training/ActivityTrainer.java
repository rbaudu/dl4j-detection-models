package com.project.training;

import com.project.common.utils.DataProcessor;
import com.project.common.utils.ImageUtils;
import com.project.models.activity.ActivityModel;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;

/**
 * Classe pour l'entraînement du modèle de détection d'activité.
 * Implémente les méthodes spécifiques pour préparer les données et gérer le modèle d'activité.
 * Utilise des images et le transfert d'apprentissage avec MobileNetV2.
 */
public class ActivityTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(ActivityTrainer.class);
    
    private final ActivityModel model;
    private final String dataDir;
    private final String modelDir;
    private final String modelName;
    private final Random random;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ActivityTrainer(Properties config) {
        super(config);
        this.model = new ActivityModel(config);
        this.dataDir = config.getProperty("activity.data.dir", "data/raw/activity");
        this.modelDir = config.getProperty("activity.model.dir", "models/activity");
        this.modelName = config.getProperty("activity.model.name", "activity_model");
        
        // Initialiser le générateur de nombres aléatoires
        int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
        this.random = new Random(seed);
        
        try {
            createDirectories(modelDir);
        } catch (IOException e) {
            log.error("Erreur lors de la création des répertoires", e);
        }
    }
    
    @Override
    protected DataSet prepareData() throws IOException {
        log.info("Préparation des données pour le modèle de détection d'activité");
        
        // Vérifier si le répertoire de données existe
        File dataDirFile = new File(dataDir);
        if (!dataDirFile.exists() || !dataDirFile.isDirectory()) {
            log.error("Le répertoire de données n'existe pas: {}", dataDir);
            throw new IOException("Répertoire de données non trouvé: " + dataDir);
        }
        
        // Configurer les dimensions d'image pour MobileNetV2
        int width = ImageUtils.MOBILENET_WIDTH;
        int height = ImageUtils.MOBILENET_HEIGHT;
        int channels = ImageUtils.MOBILENET_CHANNELS;
        
        // Configurer le générateur d'étiquettes (basé sur le nom du répertoire parent)
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        // Lister les fichiers d'images dans le répertoire
        FileSplit fileSplit = new FileSplit(dataDirFile, NativeImageLoader.ALLOWED_FORMATS, random);
        
        // Équilibrer les classes (même nombre d'exemples par classe)
        int numExamplesPerClass = Integer.parseInt(config.getProperty("activity.training.examples.per.class", "20"));
        BalancedPathFilter pathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        
        // Diviser les données en ensembles d'entraînement et de test
        InputSplit[] splits = fileSplit.sample(pathFilter, numExamplesPerClass, numExamplesPerClass);
        InputSplit trainData = splits[0];
        
        // Si aucune donnée n'a été trouvée, générer des exemples synthétiques
        if (trainData.length() == 0) {
            log.warn("Aucune donnée d'image trouvée dans {}. Utilisation de données synthétiques", dataDir);
            return generateSyntheticData(model.getNumActivityClasses());
        }
        
        log.info("Chargement de {} exemples pour l'entraînement", trainData.length());
        
        // Configurer les transformations d'image pour l'augmentation de données
        ImageTransform resize = new ResizeImageTransform(width, height);
        ImageTransform scale = new ScaleImageTransform(0.5);
        List<ImageTransform> transforms = Arrays.asList(resize, scale);
        ImageTransform pipeline = new PipelineImageTransform(transforms, false);
        
        // Configurer le RecordReader pour lire les images
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(trainData, pipeline);
        
        // Créer l'itérateur de DataSet
        int numClasses = model.getNumActivityClasses();
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        
        // Normaliser les images entre -1 et 1 (prétraitement standard pour MobileNetV2)
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(-1, 1);
        iterator.setPreProcessor(preProcessor);
        
        // Combiner tous les batchs en un seul DataSet
        DataSet allData = new DataSet();
        while (iterator.hasNext()) {
            allData.add(iterator.next());
        }
        
        log.info("Données préparées: {} exemples, {} caractéristiques, {} classes", 
                allData.numExamples(), allData.getFeatures().size(1), numClasses);
        
        return allData;
    }
    
    /**
     * Génère des données synthétiques pour l'entraînement.
     * Cette méthode est utilisée uniquement lorsqu'aucune donnée réelle n'est disponible.
     *
     * @param numClasses Nombre de classes d'activité
     * @return DataSet contenant les données synthétiques
     */
    private DataSet generateSyntheticData(int numClasses) {
        log.info("Génération de données d'images synthétiques pour {} classes d'activité", numClasses);
        
        // Dimensions pour MobileNetV2
        int height = ImageUtils.MOBILENET_HEIGHT;
        int width = ImageUtils.MOBILENET_WIDTH;
        int channels = ImageUtils.MOBILENET_CHANNELS;
        int numExamples = 100; // 5 exemples par classe (pour 20 classes)
        
        // Créer un DataSet synthétique avec des images aléatoires
        DataSet syntheticData = DataProcessor.createSyntheticImageDataSet(
                numExamples, height, width, channels, numClasses, random);
        
        log.info("Données synthétiques générées: {} exemples", syntheticData.numExamples());
        
        return syntheticData;
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        // Initialiser un nouveau modèle pour le transfert d'apprentissage
        model.initNewModel();
        return model.getNetwork();
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        log.info("Sauvegarde du modèle final vers {}", modelPath);
        model.saveModel(modelPath);
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        String checkpointPath = new File(modelDir + "/checkpoints", 
                                         modelName + "_epoch_" + epoch + ".zip").getPath();
        log.info("Sauvegarde du checkpoint à l'époque {} vers {}", epoch, checkpointPath);
        model.getNetwork().save(new File(checkpointPath), true);
    }
}
