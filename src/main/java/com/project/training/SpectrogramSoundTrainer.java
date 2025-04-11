package com.project.training;

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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Implémentation de l'entraîneur pour les sons basés sur spectrogrammes
 */
public class SpectrogramSoundTrainer extends SoundTrainer {
    private static final Logger logger = LoggerFactory.getLogger(SpectrogramSoundTrainer.class);
    
    private int spectrogramHeight;
    private int spectrogramWidth;
    private String architecture;
    
    /**
     * Constructeur avec configuration
     */
    public SpectrogramSoundTrainer(Properties config) {
        super(config);
        
        // Paramètres spécifiques aux spectrogrammes
        this.spectrogramHeight = Integer.parseInt(config.getProperty("sound.model.spectrogram.height", "224"));
        this.spectrogramWidth = Integer.parseInt(config.getProperty("sound.model.spectrogram.width", "224"));
        this.architecture = config.getProperty("sound.model.architecture", "STANDARD");
        
        // Définir le type d'entraîneur selon l'architecture
        if ("VGG16".equalsIgnoreCase(architecture)) {
            this.trainerType = SoundTrainerType.SPECTROGRAM_VGG16;
        } else if ("ResNet".equalsIgnoreCase(architecture)) {
            this.trainerType = SoundTrainerType.SPECTROGRAM_RESNET;
        } else {
            this.trainerType = SoundTrainerType.SPECTROGRAM;
        }
        
        logger.info("Initialisation de l'entraîneur Spectrogram avec architecture {} et dimensions {}x{}", 
                   architecture, spectrogramHeight, spectrogramWidth);
    }
    
    /**
     * Constructeur avec architecture spécifiée
     */
    public SpectrogramSoundTrainer(Properties config, String architecture) {
        this(config);
        this.architecture = architecture;
        
        // Mettre à jour le type selon l'architecture
        if ("VGG16".equalsIgnoreCase(architecture)) {
            this.trainerType = SoundTrainerType.SPECTROGRAM_VGG16;
        } else if ("ResNet".equalsIgnoreCase(architecture)) {
            this.trainerType = SoundTrainerType.SPECTROGRAM_RESNET;
        }
        
        logger.info("Architecture forcée à: {}", architecture);
    }
    
    @Override
    public void initializeModel() {
        logger.info("Initialisation du modèle Spectrogram avec architecture {}", architecture);
        model = createModel();
    }
    
    @Override
    protected MultiLayerNetwork createModel() {
        logger.info("Création du modèle Spectrogram avec architecture {}", architecture);
        
        if ("VGG16".equalsIgnoreCase(architecture)) {
            return createVGG16Model();
        } else if ("ResNet".equalsIgnoreCase(architecture)) {
            return createResNetModel();
        } else {
            return createStandardModel();
        }
    }
    
    /**
     * Crée un modèle CNN standard pour les spectrogrammes
     */
    private MultiLayerNetwork createStandardModel() {
        int channels = 1; // spectrogrammes en niveaux de gris
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .l2(1e-5)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(hiddenLayerSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(spectrogramHeight, spectrogramWidth, channels))
                .build();
        
        // Créer et initialiser le réseau
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
    
    /**
     * Crée un modèle VGG16 simplifié
     */
    private MultiLayerNetwork createVGG16Model() {
        int channels = 1; // spectrogrammes en niveaux de gris
        
        // Version simplifiée de VGG16 (pas tous les blocs)
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .l2(1e-5)
                .list()
                // Bloc 1
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Bloc 2
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Couches fully connected
                .layer(6, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nOut(4096)
                        .activation(Activation.RELU)
                        .build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(spectrogramHeight, spectrogramWidth, channels))
                .build();
        
        // Créer et initialiser le réseau
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
    
    /**
     * Crée un modèle ResNet simplifié
     */
    private MultiLayerNetwork createResNetModel() {
        int channels = 1; // spectrogrammes en niveaux de gris
        
        // Version simplifiée de ResNet (juste quelques couches, pas de blocs résiduels)
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .l2(1e-5)
                .list()
                // Couche d'entrée
                .layer(0, new ConvolutionLayer.Builder(7, 7)
                        .nIn(channels)
                        .nOut(64)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                // Quelques couches conv
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .nOut(256)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .kernelSize(7, 7)
                        .stride(1, 1)
                        .build())
                // Couche fully connected
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(spectrogramHeight, spectrogramWidth, channels))
                .build();
        
        // Créer et initialiser le réseau
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
    
    @Override
    protected void preprocessData() {
        logger.info("Prétraitement des données pour l'entraînement sur spectrogrammes");
        // TODO: Implémenter le prétraitement des données audio en spectrogrammes
    }
}