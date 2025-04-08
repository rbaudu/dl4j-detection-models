package com.project.common.utils;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Classe utilitaire pour la création, la sauvegarde et le chargement des modèles DL4J.
 * Fournit des méthodes communes utilisées par tous les types de modèles de détection.
 */
public class ModelUtils {
    private static final Logger log = LoggerFactory.getLogger(ModelUtils.class);
    
    private ModelUtils() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Crée un réseau de neurones dense (MLP) basé sur les paramètres fournis.
     * 
     * @param seed Graine pour la génération aléatoire
     * @param inputSize Taille de la couche d'entrée
     * @param hiddenLayers Nombre de couches cachées
     * @param hiddenSize Taille des couches cachées
     * @param outputSize Taille de la couche de sortie
     * @param learningRate Taux d'apprentissage
     * @param updaterType Type d'updater ("adam", "nesterov", "rmsprop")
     * @param useRegularization Utiliser la régularisation L2
     * @param l2 Valeur de la régularisation L2
     * @param dropoutRate Taux de dropout
     * @return MultiLayerNetwork configuré
     */
    public static MultiLayerNetwork createDenseNetwork(int seed, int inputSize, int hiddenLayers, 
                                                 int hiddenSize, int outputSize, double learningRate,
                                                 String updaterType, boolean useRegularization, 
                                                 double l2, double dropoutRate) {
        
        log.info("Création d'un réseau dense avec {} entrées, {} couches cachées de taille {}, et {} sorties",
                inputSize, hiddenLayers, hiddenSize, outputSize);
        
        // Sélectionner l'updater en fonction du type spécifié
        IUpdater updater;
        switch (updaterType.toLowerCase()) {
            case "adam":
                updater = new Adam(learningRate);
                break;
            case "nesterov":
                updater = new Nesterovs(learningRate);
                break;
            case "rmsprop":
                updater = new RmsProp(learningRate);
                break;
            default:
                log.warn("Type d'updater inconnu: {}, utilisation d'Adam", updaterType);
                updater = new Adam(learningRate);
        }
        
        // Configurer le builder du réseau
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(updater)
                .weightInit(WeightInit.XAVIER);
        
        if (useRegularization) {
            builder.l2(l2);
        }
        
        // Construire le réseau en ajoutant les couches
        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        
        // Ajouter la première couche cachée
        listBuilder.layer(0, new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(hiddenSize)
                .activation(Activation.RELU)
                .dropOut(dropoutRate)
                .build());
        
        // Ajouter les couches cachées supplémentaires
        for (int i = 1; i < hiddenLayers; i++) {
            listBuilder.layer(i, new DenseLayer.Builder()
                    .nIn(hiddenSize)
                    .nOut(hiddenSize)
                    .activation(Activation.RELU)
                    .dropOut(dropoutRate)
                    .build());
        }
        
        // Ajouter la couche de sortie
        listBuilder.layer(hiddenLayers, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(hiddenSize)
                .nOut(outputSize)
                .activation(Activation.SOFTMAX)
                .build());
        
        // Configurer l'entrée et construire le réseau
        listBuilder.setInputType(InputType.feedForward(inputSize));
        MultiLayerConfiguration configuration = listBuilder.build();
        
        return new MultiLayerNetwork(configuration);
    }
    
    /**
     * Sauvegarde un modèle dans un fichier.
     * 
     * @param model Modèle à sauvegarder
     * @param filePath Chemin où sauvegarder le modèle
     * @throws IOException en cas d'erreur lors de la sauvegarde
     */
    public static void saveModel(MultiLayerNetwork model, String filePath) throws IOException {
        // Assurer que le répertoire existe
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            if (!parentDir.mkdirs()) {
                throw new IOException("Impossible de créer le répertoire pour la sauvegarde du modèle: " + parentDir);
            }
        }
        
        // Sauvegarder le modèle
        ModelSerializer.writeModel(model, filePath, true);
        log.info("Modèle sauvegardé avec succès dans {}", filePath);
    }
    
    /**
     * Charge un modèle depuis un fichier.
     * 
     * @param filePath Chemin du fichier du modèle
     * @return Le modèle chargé
     * @throws IOException en cas d'erreur lors du chargement
     */
    public static MultiLayerNetwork loadModel(String filePath) throws IOException {
        File modelFile = new File(filePath);
        if (!modelFile.exists()) {
            throw new IOException("Le fichier du modèle n'existe pas: " + filePath);
        }
        
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        log.info("Modèle chargé avec succès depuis {}", filePath);
        return model;
    }
    
    /**
     * Exporte un modèle au format zip pour DL4J.
     * 
     * @param model Modèle à exporter
     * @param exportPath Chemin où exporter le modèle
     * @param includePreProcessing Inclure les informations de prétraitement
     * @param compressionLevel Niveau de compression (0-9)
     * @throws IOException en cas d'erreur lors de l'exportation
     */
    public static void exportModelForDL4J(MultiLayerNetwork model, String exportPath, 
                                   boolean includePreProcessing, int compressionLevel) throws IOException {
        // Assurer que le répertoire existe
        File file = new File(exportPath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            if (!parentDir.mkdirs()) {
                throw new IOException("Impossible de créer le répertoire pour l'exportation du modèle: " + parentDir);
            }
        }
        
        // Exporter le modèle
        ModelSerializer.writeModel(model, exportPath, includePreProcessing, compressionLevel);
        log.info("Modèle exporté avec succès pour DL4J dans {}", exportPath);
    }
    
    /**
     * Charge les paramètres de modèle à partir des propriétés de configuration.
     * 
     * @param config Propriétés de configuration
     * @param modelType Type de modèle ("presence", "activity", "sound")
     * @return Modèle construit selon les paramètres
     */
    public static MultiLayerNetwork createModelFromConfig(Properties config, String modelType) {
        // Paramètres communs
        int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
        boolean useRegularization = Boolean.parseBoolean(config.getProperty("training.use.regularization", "true"));
        double l2 = Double.parseDouble(config.getProperty("training.l2", "0.0001"));
        double dropout = Double.parseDouble(config.getProperty("training.dropout", "0.5"));
        String updater = config.getProperty("training.updater", "adam");
        
        // Paramètres spécifiques au modèle
        String prefix = modelType.toLowerCase() + ".model.";
        int inputSize = Integer.parseInt(config.getProperty(prefix + "input.size", "64"));
        int hiddenLayers = Integer.parseInt(config.getProperty(prefix + "hidden.layers", "2"));
        int hiddenSize = Integer.parseInt(config.getProperty(prefix + "hidden.size", "128"));
        double learningRate = Double.parseDouble(config.getProperty(prefix + "learning.rate", "0.001"));
        
        // Nombre de classes de sortie (binaire par défaut)
        int outputSize = 2;
        
        // Créer et retourner le modèle
        return createDenseNetwork(seed, inputSize, hiddenLayers, hiddenSize, outputSize, 
                learningRate, updater, useRegularization, l2, dropout);
    }
}
