package com.project.training;

import com.project.common.utils.LoggingUtils;
import com.project.common.utils.ModelUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
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

/**
 * Entraîneur pour les modèles de classification de sons.
 * Permet d'entraîner des modèles à reconnaître différents types de sons.
 */
public class SoundTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(SoundTrainer.class);
    
    private int inputLength;
    private int numMfcc;
    private int numClasses;
    private int spectrogramHeight;
    private int spectrogramWidth;
    private int channels;
    private boolean useSpectrogramModel;
    
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
        this.useSpectrogramModel = false;
        this.spectrogramHeight = 224;
        this.spectrogramWidth = 224;
        this.channels = 1;
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
        this.numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "8"));
        
        // Déterminer si on utilise un modèle de spectrogramme
        String modelType = config.getProperty("sound.model.type", "STANDARD");
        this.useSpectrogramModel = "SPECTROGRAM".equalsIgnoreCase(modelType);
        
        // Configurer les dimensions du spectrogramme si nécessaire
        if (useSpectrogramModel) {
            this.spectrogramHeight = Integer.parseInt(config.getProperty("sound.spectrogram.height", "224"));
            this.spectrogramWidth = Integer.parseInt(config.getProperty("sound.spectrogram.width", "224"));
            this.channels = 1; // Spectrogrammes généralement en niveaux de gris
        }
        
        // Journaliser les paramètres d'entraînement
        LoggingUtils.logSoundTrainingParameters(config);
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
            // Créer un exemple avec des valeurs aléatoires selon le type de modèle
            DataSet example = useSpectrogramModel ? 
                createRandomSpectrogramExample(spectrogramHeight, spectrogramWidth, channels, numClasses) :
                createRandomMFCCExample(inputLength, numMfcc, numClasses);
            dataList.add(example);
        }
        
        // Fusionner tous les exemples en un seul DataSet
        DataSet allData = DataSet.merge(dataList);
        allData.shuffle();
        
        // Diviser en ensembles d'entraînement et de test
        return splitDataset(allData, trainTestRatio);
    }
    
    /**
     * Crée un exemple aléatoire de caractéristiques MFCC pour les simulations
     * @param inputLength Longueur de la séquence d'entrée
     * @param numMfcc Nombre de coefficients MFCC
     * @param numClasses Nombre de classes
     * @return Un DataSet contenant l'exemple
     */
    private DataSet createRandomMFCCExample(int inputLength, int numMfcc, int numClasses) {
        // Créer des données d'entrée 2D (format [batch=1, inputLength*numMfcc])
        INDArray input = Nd4j.rand(new int[] {1, inputLength * numMfcc});
        
        // Créer une étiquette aléatoire (one-hot encoding)
        int labelIndex = (int) (Math.random() * numClasses);
        INDArray label = Nd4j.zeros(1, numClasses);
        label.putScalar(0, labelIndex, 1.0);
        
        // Créer et retourner un DataSet
        return new DataSet(input, label);
    }
    
    /**
     * Crée un exemple aléatoire de spectrogramme pour les simulations
     * @param height Hauteur du spectrogramme
     * @param width Largeur du spectrogramme
     * @param channels Nombre de canaux (généralement 1 pour les spectrogrammes en niveaux de gris)
     * @param numClasses Nombre de classes
     * @return Un DataSet contenant l'exemple
     */
    private DataSet createRandomSpectrogramExample(int height, int width, int channels, int numClasses) {
        // Créer des données d'entrée 4D (format [batch=1, channels, height, width])
        INDArray input = Nd4j.rand(new int[] {1, channels, height, width});
        
        // Créer une étiquette aléatoire (one-hot encoding)
        int labelIndex = (int) (Math.random() * numClasses);
        INDArray label = Nd4j.zeros(1, numClasses);
        label.putScalar(0, labelIndex, 1.0);
        
        // Créer et retourner un DataSet
        return new DataSet(input, label);
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
        initializeModel();

        // Entraîner le modèle
        train(trainData, testData);
    }
    
    /**
     * Initialise le modèle approprié selon le type (MFCC ou Spectrogramme)
     */
    public void initializeModel() {
        if (useSpectrogramModel) {
            initializeSpectrogramModel();
        } else {
            initializeMFCCModel();
        }
    }
    
    /**
     * Initialise un modèle pour traiter les caractéristiques MFCC
     */
    private void initializeMFCCModel() {
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
        int printIterations = Integer.parseInt(config.getProperty("training.print.iterations", "10"));
        model.setListeners(new ScoreIterationWithLoggingListener(printIterations));
    }
    
    /**
     * Initialise un modèle CNN pour traiter les spectrogrammes
     */
    private void initializeSpectrogramModel() {
        log.info("Initialisation d'un modèle CNN pour la classification de spectrogrammes");
        log.info("Dimensions du spectrogramme: {}x{}x{}, Nombre de classes: {}", 
                spectrogramHeight, spectrogramWidth, channels, numClasses);
        
        // Configuration pour un modèle CNN adapté aux spectrogrammes
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0001))
                .list()
                // Première couche de convolution
                .layer(0, new ConvolutionLayer.Builder(3, 3)
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
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sous-échantillonnage (pooling)
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Troisième couche de convolution
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sous-échantillonnage (pooling)
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Couche dense (fully connected)
                .layer(6, new DenseLayer.Builder()
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sortie
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                // Spécifier le type d'entrée: images de hauteur x largeur avec channels canaux
                .setInputType(InputType.convolutional(spectrogramHeight, spectrogramWidth, channels))
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
        
        // Enregistrer les métriques d'entraînement
        int printIterations = Integer.parseInt(config.getProperty("training.print.iterations", "10"));
        model.setListeners(new ScoreIterationWithLoggingListener(printIterations));
        
        // Log des informations sur la structure du modèle
        long numParams = model.numParams();
        int[] inputShape = {channels, spectrogramHeight, spectrogramWidth};
        LoggingUtils.logModelStructureInfo("son (spectrogramme)", numParams, inputShape, numClasses);
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
        // Initialiser le modèle si ce n'est pas déjà fait
        if (model == null) {
            initializeModel();
        }
        return model;
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = config != null ? 
            config.getProperty("sound.model.path", modelOutputPath) : 
            modelOutputPath;
        
        log.info("Sauvegarde du modèle de détection de sons vers: {}", modelPath);
        ModelUtils.saveModel(network, modelPath, true);
        log.info("Modèle sauvegardé avec succès");
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
        log.info("Sauvegarde du checkpoint d'époque {} vers: {}", epoch, checkpointPath);
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}