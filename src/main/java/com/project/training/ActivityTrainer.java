package com.project.training;

import com.project.common.utils.LoggingUtils;
import com.project.common.utils.ModelUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;

public class ActivityTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(ActivityTrainer.class);

    private int height;
    private int width;
    private int channels;
    private int numClasses;
    private Random rng;
    private boolean autoDetectClasses;

    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param height Hauteur des images
     * @param width Largeur des images
     * @param channels Nombre de canaux (RGB=3)
     * @param numClasses Nombre de classes de sortie
     */
    public ActivityTrainer(int batchSize, int numEpochs, String modelOutputPath, int height, int width, int channels, int numClasses) {
        super(batchSize, numEpochs, modelOutputPath);
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numClasses = numClasses;
        this.rng = new Random(42);
        this.autoDetectClasses = false;
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public ActivityTrainer(Properties config) {
        super(config);
        this.height = Integer.parseInt(config.getProperty("activity.image.height", "64"));
        this.width = Integer.parseInt(config.getProperty("activity.image.width", "64"));
        this.channels = Integer.parseInt(config.getProperty("activity.image.channels", "3"));
        this.numClasses = Integer.parseInt(config.getProperty("activity.model.num.classes", "27"));
        this.rng = new Random(42);
        this.autoDetectClasses = Boolean.parseBoolean(config.getProperty("activity.auto.detect.classes", "true"));
        
        // Journaliser les paramètres d'entraînement
        LoggingUtils.logActivityTrainingParameters(config);
    }

    /**
     * Prépare et charge les données d'images pour l'entrainement
     * @param dataDir Répertoire contenant les images d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareData(String dataDir, double trainTestRatio) throws IOException {
        // Créer un FileSplit pour le répertoire de données
        File dataDirFile = new File(dataDir);
        FileSplit dataSplit = new FileSplit(dataDirFile, NativeImageLoader.ALLOWED_FORMATS, rng);

        // Créer un ImageRecordReader pour lire les images
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(dataSplit);
        
        // Détection automatique du nombre de classes si activée
        if (autoDetectClasses) {
            // Récupérer le nombre de classes à partir du RecordReader
            List<String> labels = recordReader.getLabels();
            int detectedClasses = labels.size();
            
            if (detectedClasses != numClasses) {
                log.info("Détection automatique des classes: {} classes détectées dans le répertoire {}", 
                         detectedClasses, dataDir);
                log.info("Classes détectées: {}", String.join(", ", labels));
                
                if (detectedClasses < 2) {
                    log.warn("Nombre de classes détectées insuffisant ({}), utilisation de la valeur par défaut: {}", 
                             detectedClasses, numClasses);
                } else {
                    log.info("Mise à jour du nombre de classes de {} à {}", numClasses, detectedClasses);
                    this.numClasses = detectedClasses;
                    
                    // Mettre à jour la configuration si elle existe
                    if (config != null) {
                        config.setProperty("activity.model.num.classes", String.valueOf(detectedClasses));
                    }
                }
            } else {
                log.info("Nombre de classes configuré ({}) correspond au nombre de classes détectées", numClasses);
                log.info("Classes détectées: {}", String.join(", ", labels));
            }
        }

        // Créer un DataSetIterator pour convertir les images en DataSet
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);

        // Normaliser les données d'image (mettre à l'échelle les pixels entre 0 et 1)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

        // Collecter toutes les données
        List<DataSet> dataList = new ArrayList<>();
        while (iterator.hasNext()) {
            dataList.add(iterator.next());
        }

        // Fusionner tous les mini-lots en un seul DataSet
        DataSet allData = DataSet.merge(dataList);
        allData.shuffle();
        
        // Journaliser les informations sur les données
        log.info("Données chargées: {} exemples, {} classes", allData.numExamples(), numClasses);
        
        // Diviser en ensembles d'entraînement et de test
        return splitDataset(allData, trainTestRatio);
    }

    /**
     * Prépare et charge les données d'images avec augmentation de données
     * @param dataDir Répertoire contenant les images d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareDataWithAugmentation(String dataDir, double trainTestRatio) throws IOException {
        // Créer un FileSplit pour le répertoire de données
        File dataDirFile = new File(dataDir);
        FileSplit dataSplit = new FileSplit(dataDirFile, NativeImageLoader.ALLOWED_FORMATS, rng);

        // Définir des transformations pour l'augmentation de données
        ImageTransform resizeTransform = new ResizeImageTransform(height, width);
        
        // Créer une liste de transformations
        List<Pair<ImageTransform, Double>> transforms = new ArrayList<>();
        transforms.add(new Pair<>(resizeTransform, 1.0));
        
        // Créer un pipeline de transformations
        PipelineImageTransform pipeline = new PipelineImageTransform(transforms, false);
        
        // Créer un ImageRecordReader pour lire les images
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(dataSplit, pipeline);
        
        // Détection automatique du nombre de classes si activée
        if (autoDetectClasses) {
            // Récupérer le nombre de classes à partir du RecordReader
            List<String> labels = recordReader.getLabels();
            int detectedClasses = labels.size();
            
            if (detectedClasses != numClasses) {
                log.info("Détection automatique des classes: {} classes détectées dans le répertoire {}", 
                         detectedClasses, dataDir);
                log.info("Classes détectées: {}", String.join(", ", labels));
                
                if (detectedClasses < 2) {
                    log.warn("Nombre de classes détectées insuffisant ({}), utilisation de la valeur par défaut: {}", 
                             detectedClasses, numClasses);
                } else {
                    log.info("Mise à jour du nombre de classes de {} à {}", numClasses, detectedClasses);
                    this.numClasses = detectedClasses;
                    
                    // Mettre à jour la configuration si elle existe
                    if (config != null) {
                        config.setProperty("activity.model.num.classes", String.valueOf(detectedClasses));
                    }
                }
            } else {
                log.info("Nombre de classes configuré ({}) correspond au nombre de classes détectées", numClasses);
                log.info("Classes détectées: {}", String.join(", ", labels));
            }
        }

        // Créer un DataSetIterator pour convertir les images en DataSet
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);

        // Normaliser les données d'image (mettre à l'échelle les pixels entre 0 et 1)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

        // Collecter toutes les données
        List<DataSet> dataList = new ArrayList<>();
        while (iterator.hasNext()) {
            dataList.add(iterator.next());
        }

        // Fusionner tous les mini-lots en un seul DataSet
        DataSet allData = DataSet.merge(dataList);
        allData.shuffle();
        
        // Journaliser les informations sur les données
        log.info("Données chargées avec augmentation: {} exemples, {} classes", allData.numExamples(), numClasses);

        // Diviser en ensembles d'entraînement et de test
        return splitDataset(allData, trainTestRatio);
    }

    /**
     * Initialise et entraîne le modèle pour la classification d'activité
     * @param dataDir Répertoire contenant les images d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnActivityData(String dataDir, double trainTestRatio) throws IOException {
        // Préparer les données
        DataSet[] data = prepareData(dataDir, trainTestRatio);
        DataSet trainData = data[0];
        DataSet testData = data[1];

        // Initialiser le modèle
        initializeModel(numClasses);

        // Entraîner le modèle
        train(trainData, testData);
    }
    
    /**
     * Initialise le modèle CNN pour la classification d'images
     * @param numOutputs Nombre de classes en sortie
     */
    public void initializeModel(int numOutputs) {
        // Créer un modèle CNN simple adapté aux images
        double learningRate = Double.parseDouble(config.getProperty("activity.model.learning.rate", "0.0005"));
        int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
        double l2 = Double.parseDouble(config.getProperty("training.l2", "0.0001"));
        
        log.info("Initialisation du modèle CNN pour la détection d'activité");
        log.info("Dimensions des images: {}x{}x{}", height, width, channels);
        log.info("Nombre de classes: {}", numOutputs);
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(l2)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                // Première couche de convolution
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sous-échantillonnage (pooling)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Deuxième couche de convolution
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sous-échantillonnage (pooling)
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Couche dense (fully connected)
                .layer(4, new DenseLayer.Builder()
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sortie
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                // Spécifier le type d'entrée: images de hauteur x largeur avec channels canaux
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
        
        // Enregistrer les métriques d'entraînement toutes les 10 itérations
        int printIterations = Integer.parseInt(config.getProperty("training.print.iterations", "10"));
        model.setListeners(new ScoreIterationWithLoggingListener(printIterations));
        
        // Log des informations sur la structure du modèle
        long numParams = model.numParams();
        int[] inputShape = {channels, height, width};
        LoggingUtils.logModelStructureInfo("activité", numParams, inputShape, numOutputs);
    }
    
    @Override
    protected DataSet prepareData() throws IOException {
        // Utiliser le répertoire spécifié dans la configuration
        String dataDir = config.getProperty("activity.data.dir", "data/activity");
        
        // Préparer les données sans augmentation
        DataSet[] datasets = prepareData(dataDir, 0.8);
        
        // Retourner l'ensemble complet
        return DataSet.merge(Arrays.asList(datasets));
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        // Initialiser un modèle CNN si ce n'est pas déjà fait
        if (model == null) {
            initializeModel(numClasses);
        }
        return model;
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = config != null ? 
            config.getProperty("activity.model.path", modelOutputPath) : 
            modelOutputPath;
        
        log.info("Sauvegarde du modèle de détection d'activité vers: {}", modelPath);
        ModelUtils.saveModel(network, modelPath, true);
        log.info("Modèle sauvegardé avec succès");
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        // Déterminer le chemin du checkpoint
        String baseDir = config != null ? 
            config.getProperty("activity.checkpoint.dir", "checkpoints/activity") : 
            "checkpoints/activity";
        
        // Assurer que le répertoire existe
        createDirectories(baseDir);
        
        // Créer le chemin complet
        String checkpointPath = baseDir + "/activity_model_epoch_" + epoch + ".zip";
        
        // Sauvegarder le checkpoint
        log.info("Sauvegarde du checkpoint d'époque {} vers: {}", epoch, checkpointPath);
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}