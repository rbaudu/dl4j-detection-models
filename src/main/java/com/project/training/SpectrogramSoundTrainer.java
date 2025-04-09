package com.project.training;

import com.project.common.utils.DataLoaderUtils;
import com.project.common.utils.LoggingUtils;
import com.project.common.utils.ModelUtils;
import com.project.common.utils.SpectrogramUtils;
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
 * Classe pour l'entraînement des modèles de reconnaissance sonore basés sur spectrogrammes.
 * Cette classe convertit les fichiers audio en spectrogrammes et utilise un CNN pour la classification.
 */
public class SpectrogramSoundTrainer extends BaseSoundTrainer {
    private static final Logger log = LoggerFactory.getLogger(SpectrogramSoundTrainer.class);
    
    private int spectrogramHeight;
    private int spectrogramWidth;
    private int channels;
    private String modelArchitecture;
    
    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param numClasses Nombre de classes de sortie
     * @param spectrogramHeight Hauteur du spectrogramme
     * @param spectrogramWidth Largeur du spectrogramme
     * @param channels Nombre de canaux (généralement 1 pour les spectrogrammes)
     * @param modelArchitecture Architecture du modèle (VGG16, ResNet)
     */
    public SpectrogramSoundTrainer(int batchSize, int numEpochs, String modelOutputPath, int numClasses,
                                  int spectrogramHeight, int spectrogramWidth, int channels, String modelArchitecture) {
        super(batchSize, numEpochs, modelOutputPath, numClasses, true);
        this.spectrogramHeight = spectrogramHeight;
        this.spectrogramWidth = spectrogramWidth;
        this.channels = channels;
        this.modelArchitecture = modelArchitecture;
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public SpectrogramSoundTrainer(Properties config) {
        super(config);
        this.spectrogramHeight = Integer.parseInt(config.getProperty("sound.spectrogram.height", "224"));
        this.spectrogramWidth = Integer.parseInt(config.getProperty("sound.spectrogram.width", "224"));
        this.channels = 1; // Spectrogrammes généralement en niveaux de gris
        this.modelArchitecture = config.getProperty("sound.model.architecture", "VGG16");
    }
    
    @Override
    public void initializeModel() {
        initializeSpectrogramModel();
    }
    
    /**
     * Initialise un modèle CNN pour traiter les spectrogrammes
     */
    public void initializeSpectrogramModel() {
        log.info("Initialisation d'un modèle CNN pour la classification de spectrogrammes");
        log.info("Dimensions du spectrogramme: {}x{}x{}, Nombre de classes: {}", 
                spectrogramHeight, spectrogramWidth, channels, numClasses);
        log.info("Architecture du modèle: {}", modelArchitecture);
        
        // Choisir la configuration du modèle en fonction de l'architecture
        if ("VGG16".equalsIgnoreCase(modelArchitecture)) {
            initializeVGG16Model();
        } else if ("ResNet".equalsIgnoreCase(modelArchitecture)) {
            initializeResNetModel();
        } else {
            // Par défaut, utiliser une architecture CNN simple
            initializeSimpleCNNModel();
        }
        
        // Enregistrer les métriques d'entraînement
        int printIterations = Integer.parseInt(config != null ? 
            config.getProperty("training.print.iterations", "10") : "10");
        model.setListeners(new ScoreIterationWithLoggingListener(printIterations));
        
        // Log des informations sur la structure du modèle
        long numParams = model.numParams();
        int[] inputShape = {channels, spectrogramHeight, spectrogramWidth};
        LoggingUtils.logModelStructureInfo("son (spectrogramme)", numParams, inputShape, numClasses);
    }
    
    /**
     * Initialise un modèle VGG16 simplifié pour les spectrogrammes
     */
    private void initializeVGG16Model() {
        // Configuration pour un modèle VGG16 adapté aux spectrogrammes
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0001))
                .list()
                // Bloc 1
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Bloc 2
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Bloc 3
                .layer(6, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Couches denses (fully connected)
                .layer(10, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(Activation.RELU)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(Activation.RELU)
                        .build())
                // Couche de sortie
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                // Spécifier le type d'entrée: images de hauteur x largeur avec channels canaux
                .setInputType(InputType.convolutional(spectrogramHeight, spectrogramWidth, channels))
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
    }
    
    /**
     * Initialise un modèle ResNet simplifié pour les spectrogrammes
     */
    private void initializeResNetModel() {
        // Configuration pour un modèle ResNet adapté aux spectrogrammes
        // Note: Ceci est une version simplifiée de ResNet
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0001))
                .list()
                // Première couche de convolution
                .layer(0, new ConvolutionLayer.Builder(7, 7)
                        .nIn(channels)
                        .stride(2, 2)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                // Max pooling
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                // Couches de convolution (bloc simplifié)
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                // Couches de convolution (bloc 2)
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .stride(2, 2)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                // Couches de convolution (bloc 3)
                .layer(6, new ConvolutionLayer.Builder(3, 3)
                        .stride(2, 2)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                // Global average pooling
                .layer(8, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .kernelSize(7, 7)
                        .stride(1, 1)
                        .build())
                // Couche de sortie
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                // Spécifier le type d'entrée: images de hauteur x largeur avec channels canaux
                .setInputType(InputType.convolutional(spectrogramHeight, spectrogramWidth, channels))
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
    }
    
    /**
     * Initialise un modèle CNN simple pour les spectrogrammes
     */
    private void initializeSimpleCNNModel() {
        // Configuration pour un modèle CNN standard
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
    }
    
    @Override
    protected DataSet[] prepareAudioData(String sourceDir, String basePath, double trainTestRatio) throws IOException {
        log.info("Préparation des données audio (spectrogrammes) depuis: {}", sourceDir);
        
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
        DataSet dataSet = DataLoaderUtils.createSpectrogramDataSet(audioFiles, labelMap, 
                spectrogramHeight, spectrogramWidth, channels);
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
            config.getProperty("sound.spectrogram.model.path", modelOutputPath) : 
            modelOutputPath;
        
        log.info("Sauvegarde du modèle Spectrogram de détection de sons vers: {}", modelPath);
        ModelUtils.saveModel(network, modelPath, true);
        log.info("Modèle sauvegardé avec succès");
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        // Déterminer le chemin du checkpoint
        String baseDir = config != null ? 
            config.getProperty("sound.spectrogram.checkpoint.dir", "checkpoints/sound/spectrogram") : 
            "checkpoints/sound/spectrogram";
        
        // Assurer que le répertoire existe
        createDirectories(baseDir);
        
        // Créer le chemin complet
        String checkpointPath = baseDir + "/sound_spectrogram_model_epoch_" + epoch + ".zip";
        
        // Sauvegarder le checkpoint
        log.info("Sauvegarde du checkpoint d'époque {} vers: {}", epoch, checkpointPath);
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}
