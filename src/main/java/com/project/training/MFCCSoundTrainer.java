package com.project.training;

import com.project.common.utils.LoggingUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
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

public class MFCCSoundTrainer extends SoundTrainer {
    private static final Logger logger = LoggerFactory.getLogger(MFCCSoundTrainer.class);
    
    private int inputLength;
    private int numMfcc;
    private int inputSize;
    
    /**
     * Constructeur avec configuration
     */
    public MFCCSoundTrainer(Properties config) {
        super(config);
        
        // Charger les paramètres spécifiques aux MFCC
        inputLength = Integer.parseInt(config.getProperty("sound.input.length", "16000"));
        numMfcc = Integer.parseInt(config.getProperty("sound.num.mfcc", "40"));
        
        // Calculer la taille d'entrée pour le réseau neuronal
        // Pour une extraction MFCC, la taille d'entrée est généralement différente
        // de l'inputLength original; nous utilisons une taille fixe pour le réseau
        inputSize = 512; // Taille fixe
        
        logger.info("Configuration MFCC chargée: inputLength={}, numMfcc={}", inputLength, numMfcc);
    }
    
    @Override
    protected MultiLayerNetwork createModel() {
        logger.info("Initialisation d'un modèle MFCC pour la classification de sons");
        logger.info("Nombre d'entrées: {}, Nombre de classes: {}", inputSize, numClasses);
        
        // Configuration du réseau pour la classification audio MFCC
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(hiddenLayerSize)
                .activation(Activation.RELU)
                .dropOut(0.3)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(hiddenLayerSize)
                .nOut(hiddenLayerSize)
                .activation(Activation.RELU)
                .dropOut(0.3)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nIn(hiddenLayerSize)
                .nOut(hiddenLayerSize / 2)
                .activation(Activation.RELU)
                .dropOut(0.3)
                .build())
            .layer(3, new DenseLayer.Builder()
                .nIn(hiddenLayerSize / 2)
                .nOut(hiddenLayerSize / 4)
                .activation(Activation.RELU)
                .dropOut(0.3)
                .build())
            .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(hiddenLayerSize / 4)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.feedForward(inputSize))
            .backpropType(BackpropType.Standard)
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        
        // Afficher les informations sur le modèle
        LoggingUtils.logModelInfo(model, "son", "MFCC");
        
        return model;
    }
    
    @Override
    protected void preprocessData() {
        // Implémentation de prétraitement MFCC
        logger.info("Prétraitement des données audio avec extraction MFCC");
        // TODO: Implémenter l'extraction de caractéristiques MFCC
    }
    
    /**
     * Getter pour la taille d'entrée
     */
    public int getInputSize() {
        return inputSize;
    }
}