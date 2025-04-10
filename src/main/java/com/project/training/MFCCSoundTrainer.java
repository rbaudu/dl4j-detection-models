package com.project.training;

import com.project.common.utils.AudioUtils;
import com.project.common.utils.DataLoaderUtils;
import com.project.common.utils.ModelUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Classe pour l'entraînement des modèles de reconnaissance sonore basés sur MFCC.
 * Cette classe utilise des coefficients MFCC extraits des fichiers audio.
 */
public class MFCCSoundTrainer extends BaseSoundTrainer {
    private static final Logger log = LoggerFactory.getLogger(MFCCSoundTrainer.class);
    
    private int inputLength;
    private int numMfcc;
    
    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param numClasses Nombre de classes de sortie
     * @param inputLength Longueur de la séquence d'entrée
     * @param numMfcc Nombre de coefficients MFCC
     */
    public MFCCSoundTrainer(int batchSize, int numEpochs, String modelOutputPath, int numClasses, int inputLength, int numMfcc) {
        super(batchSize, numEpochs, modelOutputPath, numClasses, false);
        this.inputLength = inputLength;
        this.numMfcc = numMfcc;
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public MFCCSoundTrainer(Properties config) {
        super(config);
        this.inputLength = Integer.parseInt(config.getProperty("sound.input.length", "16000"));
        this.numMfcc = Integer.parseInt(config.getProperty("sound.num.mfcc", "40"));
    }
    
    @Override
    public void initializeModel() {
        initializeMFCCModel();
    }
    
    /**
     * Initialise un modèle pour traiter les caractéristiques MFCC
     */
    public void initializeMFCCModel() {
        // Configuration pour un modèle de traitement de MFCC (réseau de neurones standard)
        int numInputs = inputLength * numMfcc;
        
        log.info("Initialisation d'un modèle MFCC pour la classification de sons");
        log.info("Nombre d'entrées: {}, Nombre de classes: {}", numInputs, numClasses);
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
        
        // Enregistrer les métriques d'entraînement
        int printIterations = Integer.parseInt(config != null ? 
            config.getProperty("training.print.iterations", "10") : "10");
        model.setListeners(new org.deeplearning4j.optimize.listeners.ScoreIterationListener(printIterations));
    }
    
    @Override
    protected DataSet[] prepareAudioData(String sourceDir, String basePath, double trainTestRatio) throws IOException {
        log.info("Préparation des données audio MFCC depuis: {}", sourceDir);
        
        // Charger les fichiers audio
        List<File> audioFiles = DataLoaderUtils.loadAudioFiles(sourceDir, null);
        if (audioFiles.isEmpty()) {
            throw new IOException("Aucun fichier audio trouvé dans " + sourceDir);
        }
        
        log.info("Nombre de fichiers audio trouvés: {}", audioFiles.size());
        
        // Extraire les étiquettes des répertoires
        Map<String, Integer> labelMap = DataLoaderUtils.extractLabelsFromDirectories(sourceDir);
        if (labelMap.isEmpty()) {
            throw new IOException("Aucune classe d'activité trouvée dans " + sourceDir);
        }
        
        // Mettre à jour le nombre de classes si nécessaire
        numClasses = labelMap.size();
        log.info("Nombre de classes détectées: {}", numClasses);
        
        // Créer le DataSet à partir des fichiers audio
        DataSet dataSet = DataLoaderUtils.createMFCCDataSet(audioFiles, labelMap, numMfcc, inputLength);
        if (dataSet.numExamples() == 0) {
            throw new IOException("Échec de la création du DataSet");
        }
        
        log.info("DataSet créé avec {} exemples", dataSet.numExamples());
        
        // Diviser en ensembles d'entraînement et de test
        return DataLoaderUtils.createTrainTestSplit(dataSet, trainTestRatio);
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        // Initialiser le modèle si ce n'est pas déjà fait
        if (model == null) {
            initializeModel();
        }
        return model;
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = config != null ? 
            config.getProperty("sound.mfcc.model.path", modelOutputPath) : 
            modelOutputPath;
        
        log.info("Sauvegarde du modèle MFCC de détection de sons vers: {}", modelPath);
        ModelUtils.saveModel(network, modelPath, true);
        log.info("Modèle sauvegardé avec succès");
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        // Déterminer le chemin du checkpoint
        String baseDir = config != null ? 
            config.getProperty("sound.mfcc.checkpoint.dir", "checkpoints/sound/mfcc") : 
            "checkpoints/sound/mfcc";
        
        // Assurer que le répertoire existe
        createDirectories(baseDir);
        
        // Créer le chemin complet
        String checkpointPath = baseDir + "/sound_mfcc_model_epoch_" + epoch + ".zip";
        
        // Sauvegarder le checkpoint
        log.info("Sauvegarde du checkpoint d'époque {} vers: {}", epoch, checkpointPath);
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}
