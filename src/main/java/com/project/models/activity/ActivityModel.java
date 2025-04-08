package com.project.models.activity;

import com.project.common.utils.ModelUtils;
import com.project.common.utils.TransferLearningHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Modèle de détection d'activité.
 * Cette classe encapsule toute la logique spécifique au modèle de détection d'activité.
 * Utilise le transfert d'apprentissage avec MobileNetV2 pour la classification d'activités.
 */
public class ActivityModel {
    private static final Logger log = LoggerFactory.getLogger(ActivityModel.class);
    
    private MultiLayerNetwork network;
    private final Properties config;
    private final String modelName;
    private final String modelDir;
    private final int numActivityClasses;
    private final Map<Integer, String> labelMap;
    private final List<String> activityLabels;
    
    /**
     * Constructeur avec configuration.
     * 
     * @param config Propriétés de configuration
     */
    public ActivityModel(Properties config) {
        this.config = config;
        this.modelName = config.getProperty("activity.model.name", "activity_model");
        this.modelDir = config.getProperty("activity.model.dir", "models/activity");
        
        // Initialiser les labels des activités
        this.activityLabels = Arrays.asList(
            "CLEANING", "CONVERSING", "COOKING", "DANCING", "EATING", "FEEDING",
            "GOING_TO_SLEEP", "KNITTING", "IRONING", "LISTENING_MUSIC", "MOVING",
            "NEEDING_HELP", "PHONING", "PLAYING", "PLAYING_MUSIC", "PUTTING_AWAY",
            "READING", "RECEIVING", "SINGING", "SLEEPING", "UNKNOWN", "USING_SCREEN",
            "WAITING", "WAKING_UP", "WASHING", "WATCHING_TV", "WRITING"
        );
        
        this.numActivityClasses = activityLabels.size();
        
        // Initialiser la map des étiquettes
        this.labelMap = new HashMap<>();
        initLabelMap();
    }
    
    /**
     * Initialise la map des étiquettes pour les classes d'activité.
     */
    private void initLabelMap() {
        for (int i = 0; i < activityLabels.size(); i++) {
            labelMap.put(i, activityLabels.get(i));
        }
        log.info("Map des étiquettes d'activités initialisée avec {} classes", labelMap.size());
    }
    
    /**
     * Initialise un nouveau modèle basé sur la configuration.
     * Utilise le transfert d'apprentissage avec MobileNetV2.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle de détection d'activité avec {} classes", numActivityClasses);
        
        try {
            // Charger les paramètres depuis la configuration
            int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
            double learningRate = Double.parseDouble(config.getProperty("activity.model.learning.rate", "0.0005"));
            
            // Charger MobileNetV2 et l'adapter pour notre tâche de classification d'activités
            this.network = TransferLearningHelper.loadMobileNetV2ForActivityClassification(
                    numActivityClasses, seed, learningRate);
            
            log.info("Modèle de détection d'activité initialisé avec succès par transfert d'apprentissage");
            
        } catch (IOException e) {
            log.error("Erreur lors du chargement du modèle pré-entraîné", e);
            log.info("Initialisation d'un modèle standard en fallback");
            
            // Fallback: créer un modèle standard si le transfert d'apprentissage échoue
            this.network = ModelUtils.createModelFromConfig(config, "activity");
            this.network.init();
        }
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
     * Obtient l'étiquette textuelle pour une classe d'activité.
     * 
     * @param classIndex Indice de la classe
     * @return Étiquette de la classe
     */
    public String getLabelForClass(int classIndex) {
        return labelMap.get(classIndex);
    }
    
    /**
     * Prédit la classe d'activité la plus probable et retourne son étiquette.
     * 
     * @param input Données d'entrée
     * @return Étiquette de la classe d'activité la plus probable
     */
    public String predictLabel(double[] input) {
        int classIndex = predictClass(input);
        return getLabelForClass(classIndex);
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
    
    /**
     * Obtient la liste des étiquettes d'activité.
     * 
     * @return Liste des étiquettes d'activité
     */
    public List<String> getActivityLabels() {
        return activityLabels;
    }
    
    /**
     * Obtient la map des classes d'activité.
     * 
     * @return Map des indices aux étiquettes d'activité
     */
    public Map<Integer, String> getLabelMap() {
        return new HashMap<>(labelMap);
    }
}
