package com.project.training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
    private int inputSize; // Stocke explicitement la taille d'entrée
    
    /**
     * Constructeur avec configuration
     */
    public MFCCSoundTrainer(Properties config) {
        super(config);
        
        // Paramètres spécifiques aux MFCC
        this.numMfcc = Integer.parseInt(config.getProperty("sound.model.mfcc.coefficients", "40"));
        this.mfccLength = Integer.parseInt(config.getProperty("sound.model.mfcc.length", "300"));
        this.inputSize = numMfcc * mfccLength; // Calcul de la taille d'entrée
        this.trainerType = SoundTrainerType.MFCC;
        
        logger.info("Initialisation de l'entraîneur MFCC avec {} coefficients et longueur {}", numMfcc, mfccLength);
        logger.info("Taille d'entrée du modèle MFCC: {}", inputSize);
    }
    
    @Override
    public void initializeModel() {
        logger.info("Initialisation du modèle MFCC");
        model = createModel();
        
        // Vérifier la taille d'entrée après initialisation
        int actualInputSize = model.getLayer(0).getParam("W").columns();
        logger.info("Taille d'entrée réelle après initialisation: {}", actualInputSize);
        
        // Si la taille d'entrée ne correspond pas, forcer manuellement la taille correcte
        if (actualInputSize != inputSize) {
            logger.warn("La taille d'entrée ne correspond pas. Correction manuelle de {} à {}", actualInputSize, inputSize);
            fixInputLayerSize(model);
        }
    }
    
    /**
     * Ajuste manuellement la taille de la couche d'entrée si nécessaire
     */
    private void fixInputLayerSize(MultiLayerNetwork model) {
        try {
            // Obtenir les paramètres actuels
            INDArray currentW = model.getLayer(0).getParam("W");
            INDArray currentB = model.getLayer(0).getParam("b");
            
            // Créer une nouvelle matrice W avec la bonne taille d'entrée
            int outputSize = currentW.rows();
            INDArray newW = Nd4j.randn(outputSize, inputSize).muli(0.1);
            
            // Remplacer les paramètres du modèle
            model.getLayer(0).setParam("W", newW);
            
            logger.info("Taille d'entrée corrigée: W shape = {}", newW.shape());
        } catch (Exception e) {
            logger.error("Erreur lors de la correction de la taille d'entrée", e);
        }
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
        
        // Vérifier la taille d'entrée juste après l'initialisation
        int actualInputSize = model.getLayer(0).getParam("W").columns();
        logger.info("Taille d'entrée après initialisation dans createModel: {}", actualInputSize);
        
        return model;
    }
    
    @Override
    protected void preprocessData() {
        logger.info("Prétraitement des données pour l'entraînement MFCC");
        // TODO: Implémenter le prétraitement des données audio en MFCC
    }
    
    // Getter pour la taille d'entrée
    public int getInputSize() {
        return inputSize;
    }
}