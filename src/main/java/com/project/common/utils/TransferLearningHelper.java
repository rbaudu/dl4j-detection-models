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
import java.util.Map;

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
        // Correction: Utiliser la méthode correcte pour obtenir les types d'entrée
        Map<String, INDArray> inputArrays = baseModel.getInputs();
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
}