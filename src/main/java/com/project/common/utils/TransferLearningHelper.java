package com.project.common.utils;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Classe utilitaire pour le transfer learning.
 * Fournit des méthodes pour configurer et utiliser des modèles pré-entraînés.
 */
public class TransferLearningHelper {

    /**
     * Configure un modèle de transfert d'apprentissage à partir d'un modèle pré-entraîné
     * @param baseModel Le modèle pré-entraîné à utiliser comme base
     * @param numOutputs Le nombre de sorties (classes) pour le nouveau modèle
     * @param freezeUntilLayer Le nom de la couche jusqu'à laquelle les poids seront gelés
     * @return Un modèle configuré pour le transfert d'apprentissage
     */
    public static ComputationGraph configureTransferLearning(ComputationGraph baseModel, int numOutputs, String freezeUntilLayer) {
        
        // Obtenir les types d'entrée du réseau
        // Création d'un Map pour stocker les entrées
        Map<String, INDArray> inputArrays = new HashMap<>();
        for (String input : baseModel.getConfiguration().getNetworkInputs()) {
            inputArrays.put(input, null); // Les valeurs réelles ne sont pas importantes ici
        }
        
        String[] inputNames = baseModel.getConfiguration().getNetworkInputs().toArray(new String[0]);
        
        // Configuration pour le fine-tuning
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(org.nd4j.linalg.learning.config.Adam.builder().learningRate(0.0001).build())
                .seed(123)
                .build();
        
        // Construire le modèle de transfert d'apprentissage
        ComputationGraph transferModel = new TransferLearning.GraphBuilder(baseModel)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(freezeUntilLayer)  // Geler jusqu'à cette couche
                .removeVertexAndConnections("output")  // Supprimer la couche de sortie existante
                .addLayer("output", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nOut(numOutputs)
                        .activation(org.nd4j.linalg.activations.Activation.SOFTMAX)
                        .lossFunction(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build(), "features")  // Connecter à la couche "features"
                .setOutputs("output")  // Définir la nouvelle couche de sortie
                .build();
        
        // Initialiser le modèle
        transferModel.init();
        
        return transferModel;
    }
    
    /**
     * Convertit un MultiLayerNetwork en ComputationGraph
     * @param network Le MultiLayerNetwork à convertir
     * @return Le ComputationGraph converti
     */
    public static ComputationGraph convertToComputationGraph(MultiLayerNetwork network) {
        ComputationGraph graph = network.toComputationGraph();
        return graph;
    }
    
    /**
     * Extrait les caractéristiques d'un modèle pré-entraîné pour une entrée donnée
     * @param model Le modèle pré-entraîné
     * @param featureExtractionLayer Le nom de la couche à partir de laquelle extraire les caractéristiques
     * @param input L'entrée pour laquelle extraire les caractéristiques
     * @return Les caractéristiques extraites
     */
    public static INDArray extractFeatures(ComputationGraph model, String featureExtractionLayer, INDArray input) {
        // Activer le mode d'extraction de caractéristiques
        model.feedForward(input, false);
        
        // Obtenir l'activation de la couche spécifiée
        Map<String, INDArray> activations = model.feedForward(input, false);
        INDArray features = activations.get(featureExtractionLayer);
        
        return features;
    }
    
    /**
     * Charge un modèle MobileNetV2 pré-entraîné et l'adapte pour la classification d'activité
     * @param inputHeight Hauteur des images d'entrée
     * @param inputWidth Largeur des images d'entrée
     * @param dropoutRate Taux de dropout à appliquer
     * @return Le modèle adapté
     */
    public static ComputationGraph loadMobileNetV2ForActivityClassification(int inputHeight, int inputWidth, double dropoutRate) {
        // Note: Dans une implémentation réelle, vous chargeriez un vrai modèle pré-entraîné
        // Ici nous créons simplement un modèle de base pour la démonstration
        
        // Créer un réseau simple pour la classification d'images
        MultiLayerNetwork baseNetwork = ModelUtils.createDeepNetwork(inputHeight * inputWidth * 3, 1000);
        
        // Convertir en ComputationGraph
        ComputationGraph baseModel = convertToComputationGraph(baseNetwork);
        
        // Adapter pour la classification d'activité (5 classes typiques)
        return configureTransferLearning(baseModel, 5, "layer2");
    }
    
    /**
     * Charge un modèle YAMNet pré-entraîné et l'adapte pour la classification de sons
     * @param inputLength Longueur de l'entrée audio
     * @param numMfcc Nombre de coefficients MFCC
     * @param dropoutRate Taux de dropout à appliquer
     * @return Le modèle adapté
     */
    public static ComputationGraph loadYAMNetForSoundClassification(int inputLength, int numMfcc, double dropoutRate) {
        // Note: Dans une implémentation réelle, vous chargeriez un vrai modèle pré-entraîné
        // Ici nous créons simplement un modèle de base pour la démonstration
        
        // Créer un réseau simple pour le traitement audio
        MultiLayerNetwork baseNetwork = ModelUtils.createDeepNetwork(inputLength * numMfcc, 1000);
        
        // Convertir en ComputationGraph
        ComputationGraph baseModel = convertToComputationGraph(baseNetwork);
        
        // Adapter pour la classification de sons (3 classes typiques)
        return configureTransferLearning(baseModel, 3, "layer2");
    }
}