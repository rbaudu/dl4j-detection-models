package com.project.models.activity;

import com.project.common.utils.ModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Modèle de détection d'activité.
 * Cette classe encapsule toute la logique spécifique au modèle de détection d'activité.
 */
public class ActivityModel {
    private static final Logger log = LoggerFactory.getLogger(ActivityModel.class);
    
    private MultiLayerNetwork network;
    private final Properties config;
    private final String modelName;
    private final String modelDir;
    private final int numActivityClasses;
    
    /**
     * Constructeur avec configuration.
     * 
     * @param config Propriétés de configuration
     */
    public ActivityModel(Properties config) {
        this.config = config;
        this.modelName = config.getProperty("activity.model.name", "activity_model");
        this.modelDir = config.getProperty("activity.model.dir", "models/activity");
        // Par défaut, on considère 4 classes d'activité: repos, marche, course, autre
        this.numActivityClasses = Integer.parseInt(config.getProperty("activity.model.num.classes", "4"));
    }
    
    /**
     * Initialise un nouveau modèle basé sur la configuration.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle de détection d'activité avec {} classes", numActivityClasses);
        
        // Obtenir les paramètres depuis la configuration
        int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
        boolean useRegularization = Boolean.parseBoolean(config.getProperty("training.use.regularization", "true"));
        double l2 = Double.parseDouble(config.getProperty("training.l2", "0.0001"));
        double dropout = Double.parseDouble(config.getProperty("training.dropout", "0.5"));
        String updater = config.getProperty("training.updater", "adam");
        
        // Paramètres spécifiques au modèle d'activité
        int inputSize = Integer.parseInt(config.getProperty("activity.model.input.size", "128"));
        int hiddenLayers = Integer.parseInt(config.getProperty("activity.model.hidden.layers", "3"));
        int hiddenSize = Integer.parseInt(config.getProperty("activity.model.hidden.size", "256"));
        double learningRate = Double.parseDouble(config.getProperty("activity.model.learning.rate", "0.0005"));
        
        // Créer le modèle avec le nombre de classes d'activité approprié
        this.network = ModelUtils.createDenseNetwork(
                seed, inputSize, hiddenLayers, hiddenSize, numActivityClasses,
                learningRate, updater, useRegularization, l2, dropout);
        
        this.network.init();
        log.info("Modèle de détection d'activité initialisé avec succès");
    }
    
    /**
     * Charge un modèle existant depuis le disque.
     * 
     * @param modelPath Chemin vers le fichier du modèle
     * @throws IOException en cas d'erreur lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle de détection d'activité depuis {}", modelPath);
        this.network = ModelUtils.loadModel(modelPath);
        log.info("Modèle chargé avec succès");
    }
    
    /**
     * Sauvegarde le modèle sur le disque.
     * 
     * @param modelPath Chemin où sauvegarder le modèle
     * @throws IOException en cas d'erreur lors de la sauvegarde
     */
    public void saveModel(String modelPath) throws IOException {
        if (this.network == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé ou chargé");
        }
        
        log.info("Sauvegarde du modèle de détection d'activité vers {}", modelPath);
        ModelUtils.saveModel(this.network, modelPath);
    }
    
    /**
     * Exporte le modèle au format DL4J pour être utilisé dans d'autres applications.
     * 
     * @param exportPath Chemin vers lequel exporter le modèle
     * @throws IOException en cas d'erreur lors de l'exportation
     */
    public void exportModel(String exportPath) throws IOException {
        if (this.network == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé ou chargé");
        }
        
        boolean includePreprocessing = Boolean.parseBoolean(
                config.getProperty("export.include.preprocessing", "true"));
        int compressionLevel = Integer.parseInt(
                config.getProperty("export.zip.compression.level", "9"));
        
        log.info("Exportation du modèle de détection d'activité vers {}", exportPath);
        ModelUtils.exportModelForDL4J(
                this.network, exportPath, includePreprocessing, compressionLevel);
        log.info("Modèle exporté avec succès");
    }
    
    /**
     * Prédit l'activité à partir des données d'entrée.
     * 
     * @param input Données d'entrée
     * @return Tableau de probabilités pour chaque classe d'activité
     */
    public double[] predict(double[] input) {
        if (this.network == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé ou chargé");
        }
        
        // Convertir l'entrée en INDArray
        INDArray inputArray = Nd4j.create(input);
        
        // Prédire
        INDArray output = this.network.output(inputArray);
        
        // Convertir la sortie en tableau Java
        return output.toDoubleVector();
    }
    
    /**
     * Prédit la classe d'activité la plus probable.
     * 
     * @param input Données d'entrée
     * @return Indice de la classe d'activité la plus probable
     */
    public int predictClass(double[] input) {
        double[] probabilities = predict(input);
        int maxIndex = 0;
        double maxProb = probabilities[0];
        
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * Obtient le réseau neuronal sous-jacent.
     * 
     * @return Le réseau neuronal
     */
    public MultiLayerNetwork getNetwork() {
        return network;
    }
    
    /**
     * Obtient le chemin par défaut du modèle sauvegardé.
     * 
     * @return Chemin par défaut du modèle
     */
    public String getDefaultModelPath() {
        return new File(modelDir, modelName + ".zip").getPath();
    }
    
    /**
     * Charge le modèle à partir du chemin par défaut.
     * 
     * @throws IOException en cas d'erreur lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = getDefaultModelPath();
        File modelFile = new File(modelPath);
        
        if (!modelFile.exists()) {
            log.warn("Modèle par défaut non trouvé à {}, initialisation d'un nouveau modèle", modelPath);
            initNewModel();
        } else {
            loadModel(modelPath);
        }
    }
    
    /**
     * Obtient le nombre de classes d'activité.
     * 
     * @return Nombre de classes d'activité
     */
    public int getNumActivityClasses() {
        return numActivityClasses;
    }
}
