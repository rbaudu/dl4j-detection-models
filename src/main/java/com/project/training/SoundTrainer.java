package com.project.training;

import com.project.common.utils.ModelUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Entraîneur pour les modèles de classification de sons.
 * Permet d'entraîner des modèles à reconnaître différents types de sons.
 */
public class SoundTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(SoundTrainer.class);
    
    private int inputLength;
    private int numMfcc;
    private int numClasses;
    
    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param inputLength Longueur de la séquence d'entrée
     * @param numMfcc Nombre de coefficients MFCC
     * @param numClasses Nombre de classes de sortie
     */
    public SoundTrainer(int batchSize, int numEpochs, String modelOutputPath, int inputLength, int numMfcc, int numClasses) {
        super(batchSize, numEpochs, modelOutputPath);
        this.inputLength = inputLength;
        this.numMfcc = numMfcc;
        this.numClasses = numClasses;
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public SoundTrainer(Properties config) {
        super(config);
        this.inputLength = Integer.parseInt(config.getProperty("sound.input.length", "16000"));
        this.numMfcc = Integer.parseInt(config.getProperty("sound.num.mfcc", "40"));
        this.numClasses = Integer.parseInt(config.getProperty("sound.num.classes", "3"));
    }

    /**
     * Prépare et charge les données audio pour l'entrainement
     * @param featureDir Répertoire contenant les fichiers de caractéristiques
     * @param labelDir Répertoire contenant les fichiers d'étiquettes
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareData(String featureDir, String labelDir, double trainTestRatio) throws IOException, InterruptedException {
        // Cette méthode simule le chargement de données audio
        // Dans une implémentation réelle, vous utiliseriez des données MFCC ou spectrales
        
        log.info("Chargement des données audio depuis {} et {}", featureDir, labelDir);
        
        // Créer une liste pour stocker les données
        List<DataSet> dataList = new ArrayList<>();
        
        // Dans cette simulation, nous créons quelques exemples synthétiques
        for (int i = 0; i < 100; i++) {
            // Créer un exemple avec des valeurs aléatoires
            DataSet example = createRandomExample(inputLength * numMfcc, numClasses);
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
     * Entraîne le modèle sur des données audio
     * @param featureDir Répertoire contenant les fichiers de caractéristiques
     * @param labelDir Répertoire contenant les fichiers d'étiquettes
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnSoundData(String featureDir, String labelDir, double trainTestRatio) throws IOException, InterruptedException {
        // Préparer les données
        DataSet[] data = prepareData(featureDir, labelDir, trainTestRatio);
        DataSet trainData = data[0];
        DataSet testData = data[1];

        // Initialiser le modèle
        int numInputs = inputLength * numMfcc;
        initializeModel(numInputs, numClasses);

        // Entraîner le modèle
        train(trainData, testData);
    }
    
    @Override
    protected DataSet prepareData() throws IOException {
        try {
            // Utiliser les répertoires spécifiés dans la configuration
            String featureDir = config.getProperty("sound.feature.dir", "data/sound/features");
            String labelDir = config.getProperty("sound.label.dir", "data/sound/labels");
            
            // Préparer les données
            DataSet[] datasets = prepareData(featureDir, labelDir, 0.8);
            
            // Retourner l'ensemble complet
            return DataSet.merge(Arrays.asList(datasets));
        } catch (InterruptedException e) {
            log.error("Erreur lors de la préparation des données audio", e);
            throw new IOException("Erreur lors de la préparation des données audio", e);
        }
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        int numInputs = inputLength * numMfcc;
        return ModelUtils.createSimpleNetwork(numInputs, numClasses);
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = config != null ? 
            config.getProperty("sound.model.path", modelOutputPath) : 
            modelOutputPath;
        
        ModelUtils.saveModel(network, modelPath, true);
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        // Déterminer le chemin du checkpoint
        String baseDir = config != null ? 
            config.getProperty("sound.checkpoint.dir", "checkpoints/sound") : 
            "checkpoints/sound";
        
        // Assurer que le répertoire existe
        createDirectories(baseDir);
        
        // Créer le chemin complet
        String checkpointPath = baseDir + "/sound_model_epoch_" + epoch + ".zip";
        
        // Sauvegarder le checkpoint
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}