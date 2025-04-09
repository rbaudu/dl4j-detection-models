package com.project.common.utils;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Classe utilitaire pour la création, la sauvegarde et le chargement de modèles.
 */
public class ModelUtils {

    /**
     * Crée un modèle de réseau de neurones simple pour la classification
     * @param numInputs Nombre d'entrées
     * @param numOutputs Nombre de sorties (classes)
     * @return Un réseau de neurones configuré
     */
    public static MultiLayerNetwork createSimpleNetwork(int numInputs, int numOutputs) {
        // Configuration du réseau
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .l2(1e-5)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        // Créer et initialiser le réseau
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
    
    /**
     * Crée un modèle de réseau de neurones profond pour la classification
     * @param numInputs Nombre d'entrées
     * @param numOutputs Nombre de sorties (classes)
     * @return Un réseau de neurones configuré
     */
    public static MultiLayerNetwork createDeepNetwork(int numInputs, int numOutputs) {
        // Configuration du réseau
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0005))
                .l2(1e-5)
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
                .layer(2, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        // Créer et initialiser le réseau
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
    
    /**
     * Crée un modèle à partir d'une configuration
     * @param config Propriétés de configuration
     * @param modelType Type de modèle (pour sélectionner les bons paramètres)
     * @return Un réseau de neurones configuré
     */
    public static MultiLayerNetwork createModelFromConfig(Properties config, String modelType) {
        String prefix = modelType + ".model.";
        int numInputs = Integer.parseInt(config.getProperty(prefix + "input.size", "64"));
        int numOutputs = Integer.parseInt(config.getProperty(prefix + "output.size", "2"));
        int numHiddenLayers = Integer.parseInt(config.getProperty(prefix + "hidden.layers", "2"));
        
        // Si on a un nombre élevé de couches cachées, utiliser un réseau profond
        if (numHiddenLayers > 2) {
            return createDeepNetwork(numInputs, numOutputs);
        } else {
            return createSimpleNetwork(numInputs, numOutputs);
        }
    }
    
    /**
     * Sauvegarde un modèle sur le disque
     * @param model Le modèle à sauvegarder
     * @param filePath Chemin où sauvegarder le modèle
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public static void saveModel(MultiLayerNetwork model, String filePath) throws IOException {
        saveModel(model, filePath, true);
    }
    
    /**
     * Sauvegarde un modèle sur le disque
     * @param model Le modèle à sauvegarder
     * @param filePath Chemin où sauvegarder le modèle
     * @param includeUpdater Indique s'il faut inclure l'état de l'optimiseur
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public static void saveModel(MultiLayerNetwork model, String filePath, boolean includeUpdater) throws IOException {
        // Correction: Utiliser File au lieu de String
        File file = new File(filePath);
        
        // Créer le répertoire parent si nécessaire
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        // Sauvegarder le modèle
        ModelSerializer.writeModel(model, file, includeUpdater);
    }
    
    /**
     * Exporte un modèle dans un format adapté pour DL4J
     * @param model Le modèle à exporter
     * @param filePath Chemin où sauvegarder le modèle
     * @param includeUpdater Indique s'il faut inclure l'état de l'optimiseur
     * @param version Version du format d'export
     * @throws IOException Si une erreur survient lors de l'export
     */
    public static void exportModelForDL4J(MultiLayerNetwork model, String filePath, boolean includeUpdater, int version) throws IOException {
        // Pour la compatibilité, utiliser simplement saveModel mais avec un log spécifique
        System.out.println("Exportation du modèle au format DL4J v" + version + " vers " + filePath);
        saveModel(model, filePath, includeUpdater);
    }
    
    /**
     * Charge un modèle depuis le disque
     * @param filePath Chemin du modèle à charger
     * @return Le modèle chargé
     * @throws IOException Si une erreur survient lors du chargement
     */
    public static MultiLayerNetwork loadModel(String filePath) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
    }
}