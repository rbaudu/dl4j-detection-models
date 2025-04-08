package com.project.models.sound;

import com.project.common.utils.ModelUtils;
import com.project.common.utils.TransferLearningHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * Modèle de détection de sons.
 * Cette classe encapsule toute la logique spécifique au modèle de détection de sons.
 * Utilise le transfert d'apprentissage avec YAMNet pour la classification de sons.
 */
public class SoundModel {
    private static final Logger log = LoggerFactory.getLogger(SoundModel.class);
    
    private MultiLayerNetwork network;
    private final Properties config;
    private final String modelName;
    private final String modelDir;
    private final int numSoundClasses;
    private final Map<Integer, String> labelMap;
    
    /**
     * Constructeur avec configuration.
     * 
     * @param config Propriétés de configuration
     */
    public SoundModel(Properties config) {
        this.config = config;
        this.modelName = config.getProperty("sound.model.name", "sound_model");
        this.modelDir = config.getProperty("sound.model.dir", "models/sound");
        
        // Par défaut, considère plusieurs classes de sons communs
        this.numSoundClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "5"));
        
        // Initialiser la map des étiquettes
        this.labelMap = new HashMap<>();
        initLabelMap();
    }
    
    /**
     * Initialise la map des étiquettes pour les classes de sons.
     */
    private void initLabelMap() {
        // Récupérer les étiquettes depuis la configuration ou utiliser des valeurs par défaut
        String labelsPrefix = "sound.model.label.";
        
        // Essayer de charger les étiquettes depuis la configuration
        boolean foundLabels = false;
        for (int i = 0; i < numSoundClasses; i++) {
            String label = config.getProperty(labelsPrefix + i);
            if (label != null) {
                labelMap.put(i, label);
                foundLabels = true;
            }
        }
        
        // Si aucune étiquette n'est trouvée dans la configuration, utiliser des valeurs par défaut
        if (!foundLabels) {
            labelMap.put(0, "Silence");
            labelMap.put(1, "Parole");
            labelMap.put(2, "Musique");
            labelMap.put(3, "Bruit ambiant");
            labelMap.put(4, "Alarme");
            
            // Ajouter des étiquettes supplémentaires si nécessaire
            if (numSoundClasses > 5) {
                for (int i = 5; i < numSoundClasses; i++) {
                    labelMap.put(i, "Son " + i);
                }
            }
        }
    }
    
    /**
     * Initialise un nouveau modèle basé sur la configuration.
     * Utilise le transfert d'apprentissage avec YAMNet.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle de détection de sons avec {} classes", numSoundClasses);
        
        try {
            // Obtenir les paramètres depuis la configuration
            int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
            double learningRate = Double.parseDouble(config.getProperty("sound.model.learning.rate", "0.0001"));
            
            // Charger YAMNet et l'adapter pour notre tâche de classification de sons
            this.network = TransferLearningHelper.loadYAMNetForSoundClassification(
                    numSoundClasses, seed, learningRate);
            
            log.info("Modèle de détection de sons initialisé avec succès par transfert d'apprentissage");
            log.info("Classes de sons configurées: {}", labelMap);
            
        } catch (IOException e) {
            log.error("Erreur lors du chargement du modèle pré-entraîné", e);
            log.info("Initialisation d'un modèle standard en fallback");
            
            // Fallback: créer un modèle standard si le transfert d'apprentissage échoue
            this.network = ModelUtils.createModelFromConfig(config, "sound");
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
        log.info("Chargement du modèle de détection de sons depuis {}", modelPath);
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
        
        log.info("Sauvegarde du modèle de détection de sons vers {}", modelPath);
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
        
        log.info("Exportation du modèle de détection de sons vers {}", exportPath);
        ModelUtils.exportModelForDL4J(
                this.network, exportPath, includePreprocessing, compressionLevel);
        log.info("Modèle exporté avec succès");
    }
    
    /**
     * Prédit la classe de son à partir des données d'entrée.
     * 
     * @param input Données d'entrée (caractéristiques audio)
     * @return Tableau de probabilités pour chaque classe de son
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
     * Prédit la classe de son la plus probable.
     * 
     * @param input Données d'entrée (caractéristiques audio)
     * @return Indice de la classe de son la plus probable
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
     * Obtient l'étiquette textuelle pour une classe de son.
     * 
     * @param classIndex Indice de la classe
     * @return Étiquette de la classe ou null si non trouvée
     */
    public String getLabelForClass(int classIndex) {
        return labelMap.get(classIndex);
    }
    
    /**
     * Prédit la classe de son la plus probable et retourne son étiquette.
     * 
     * @param input Données d'entrée (caractéristiques audio)
     * @return Étiquette de la classe de son la plus probable
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
     * Obtient le nombre de classes de sons.
     * 
     * @return Nombre de classes de sons
     */
    public int getNumSoundClasses() {
        return numSoundClasses;
    }
    
    /**
     * Obtient la map des étiquettes.
     * 
     * @return Map associant les indices de classes à leurs étiquettes
     */
    public Map<Integer, String> getLabelMap() {
        return new HashMap<>(labelMap);
    }
}
