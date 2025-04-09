package com.project.training;

import com.project.common.utils.ModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Entraîneur pour les modèles de détection de présence.
 * Permet d'entraîner des modèles à détecter la présence ou l'absence de personnes.
 */
public class PresenceTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(PresenceTrainer.class);
    
    private int inputSize;
    private int numClasses;
    
    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param inputSize Taille de l'entrée
     * @param numClasses Nombre de classes de sortie
     */
    public PresenceTrainer(int batchSize, int numEpochs, String modelOutputPath, int inputSize, int numClasses) {
        super(batchSize, numEpochs, modelOutputPath);
        this.inputSize = inputSize;
        this.numClasses = numClasses;
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public PresenceTrainer(Properties config) {
        super(config);
        this.inputSize = Integer.parseInt(config.getProperty("presence.input.size", "100"));
        this.numClasses = Integer.parseInt(config.getProperty("presence.num.classes", "2"));
    }

    /**
     * Prépare et charge les données pour l'entrainement
     * @param dataDir Répertoire contenant les données d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareData(String dataDir, double trainTestRatio) throws IOException {
        // Cette méthode simule le chargement de données de présence
        // Dans une implémentation réelle, vous utiliseriez des capteurs de présence
        
        log.info("Chargement des données de présence depuis {}", dataDir);
        
        // Créer une liste pour stocker les données
        List<DataSet> dataList = new ArrayList<>();
        
        // Dans cette simulation, nous créons quelques exemples synthétiques
        for (int i = 0; i < 100; i++) {
            // Créer un exemple avec des valeurs aléatoires
            DataSet example = createRandomExample(inputSize, numClasses);
            dataList.add(example);
        }
        
        // Fusionner tous les exemples en un seul DataSet
        DataSet allData = DataSet.merge(dataList);
        allData.shuffle();
        
        // Diviser en ensembles d'entraînement et de test
        return splitDataset(allData, trainTestRatio);
    }
    
    /**
     * Crée un exemple aléatoire pour les simulations
     * @param inputSize Taille de l'entrée
     * @param numClasses Nombre de classes
     * @return Un DataSet contenant l'exemple
     */
    private DataSet createRandomExample(int inputSize, int numClasses) {
        // Créer des données d'entrée aléatoires
        float[] input = new float[inputSize];
        for (int i = 0; i < inputSize; i++) {
            input[i] = (float) Math.random();
        }
        
        // Créer une étiquette aléatoire (one-hot encoding)
        int labelIndex = (int) (Math.random() * numClasses);
        float[] label = new float[numClasses];
        label[labelIndex] = 1.0f;
        
        // Créer et retourner un DataSet
        return new DataSet(org.nd4j.linalg.factory.Nd4j.create(input), org.nd4j.linalg.factory.Nd4j.create(label));
    }

    /**
     * Entraîne le modèle sur des données de présence
     * @param dataDir Répertoire contenant les données d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnPresenceData(String dataDir, double trainTestRatio) throws IOException {
        // Préparer les données
        DataSet[] data = prepareData(dataDir, trainTestRatio);
        DataSet trainData = data[0];
        DataSet testData = data[1];

        // Initialiser le modèle
        initializeModel(inputSize, numClasses);

        // Entraîner le modèle
        train(trainData, testData);
    }
    
    @Override
    protected DataSet prepareData() throws IOException {
        // Utiliser le répertoire spécifié dans la configuration
        String dataDir = config.getProperty("presence.data.dir", "data/presence");
        
        // Préparer les données
        DataSet[] datasets = prepareData(dataDir, 0.8);
        
        // Retourner l'ensemble complet
        return DataSet.merge(Arrays.asList(datasets));
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        return ModelUtils.createSimpleNetwork(inputSize, numClasses);
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = config != null ? 
            config.getProperty("presence.model.path", modelOutputPath) : 
            modelOutputPath;
        
        ModelUtils.saveModel(network, modelPath, true);
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        // Déterminer le chemin du checkpoint
        String baseDir = config != null ? 
            config.getProperty("presence.checkpoint.dir", "checkpoints/presence") : 
            "checkpoints/presence";
        
        // Assurer que le répertoire existe
        createDirectories(baseDir);
        
        // Créer le chemin complet
        String checkpointPath = baseDir + "/presence_model_epoch_" + epoch + ".zip";
        
        // Sauvegarder le checkpoint
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}