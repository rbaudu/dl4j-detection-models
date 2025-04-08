package com.project.models.presence;

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
 * Modèle de détection de présence.
 * Cette classe encapsule toute la logique spécifique au modèle de détection de présence.
 */
public class PresenceModel {
    private static final Logger log = LoggerFactory.getLogger(PresenceModel.class);
    
    private MultiLayerNetwork network;
    private final Properties config;
    private final String modelName;
    private final String modelDir;
    
    /**
     * Constructeur avec configuration.
     * 
     * @param config Propriétés de configuration
     */
    public PresenceModel(Properties config) {
        this.config = config;
        this.modelName = config.getProperty("presence.model.name", "presence_model");
        this.modelDir = config.getProperty("presence.model.dir", "models/presence");
    }
    
    /**
     * Initialise un nouveau modèle basé sur la configuration.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle de détection de présence");
        this.network = ModelUtils.createModelFromConfig(config, "presence");
        this.network.init();
        log.info("Modèle initialisé avec succès");
    }
    
    /**
     * Charge un modèle existant depuis le disque.
     * 
     * @param modelPath Chemin vers le fichier du modèle
     * @throws IOException en cas d'erreur lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle de détection de présence depuis {}", modelPath);
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
        
        log.info("Sauvegarde du modèle de détection de présence vers {}", modelPath);
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
        
        log.info("Exportation du modèle de détection de présence vers {}", exportPath);
        ModelUtils.exportModelForDL4J(
                this.network, exportPath, includePreprocessing, compressionLevel);
        log.info("Modèle exporté avec succès");
    }
    
    /**
     * Prédit la présence à partir des données d'entrée.
     * 
     * @param input Données d'entrée
     * @return Tableau de probabilités [probAbsence, probPresence]
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
}
