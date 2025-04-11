package com.project.training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Implémentation de l'entraîneur pour les sons basés sur MFCC
 */
public class MFCCSoundTrainer extends SoundTrainer {
    private static final Logger logger = LoggerFactory.getLogger(MFCCSoundTrainer.class);
    
    private int numMfcc;
    private int mfccLength;
    private int inputSize; // Taille d'entrée calculée
    
    /**
     * Constructeur avec configuration
     */
    public MFCCSoundTrainer(Properties config) {
        super(config);
        
        // Paramètres spécifiques aux MFCC
        this.numMfcc = Integer.parseInt(config.getProperty("sound.model.mfcc.coefficients", "40"));
        this.mfccLength = Integer.parseInt(config.getProperty("sound.model.mfcc.length", "300"));
        this.inputSize = numMfcc * mfccLength;
        this.trainerType = SoundTrainerType.MFCC;
        
        logger.info("Initialisation de l'entraîneur MFCC avec {} coefficients et longueur {}", numMfcc, mfccLength);
        logger.info("Taille d'entrée calculée: {}", inputSize);
    }
    
    @Override
    public void initializeModel() {
        logger.info("Initialisation du modèle MFCC");
        model = createModel();
        
        // Obtenir la taille d'entrée réelle du modèle après création
        int actualInputSize = model.getLayer(0).getParam("W").columns();
        logger.info("Taille d'entrée réelle du modèle après initialisation: {}", actualInputSize);
    }
    
    @Override
    protected MultiLayerNetwork createModel() {
        logger.info("Création du modèle MFCC avec taille d'entrée: {}", inputSize);
        
        // Configurer le réseau
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .l2(1e-5)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(hiddenLayerSize / 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        // Créer et initialiser le réseau
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        // Vérifier la taille d'entrée après création
        int actualInputSize = model.getLayer(0).getParam("W").columns();
        logger.info("Taille d'entrée du modèle créé: {}", actualInputSize);
        
        return model;
    }
    
    @Override
    protected void preprocessData() {
        logger.info("Prétraitement des données pour l'entraînement MFCC");
        // TODO: Implémenter le prétraitement des données audio en MFCC
    }
    
    /**
     * Retourne la taille d'entrée calculée à partir des paramètres MFCC
     * @return Taille d'entrée du modèle (numMfcc * mfccLength)
     */
    public int getInputSize() {
        return inputSize;
    }
    
    /**
     * Retourne le nombre de coefficients MFCC
     * @return Nombre de coefficients MFCC
     */
    public int getNumMfcc() {
        return numMfcc;
    }
    
    /**
     * Retourne la longueur des séquences MFCC
     * @return Longueur des séquences MFCC
     */
    public int getMfccLength() {
        return mfccLength;
    }
}