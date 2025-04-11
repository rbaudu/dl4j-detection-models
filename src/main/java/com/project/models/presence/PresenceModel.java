package com.project.models.presence;

import com.project.common.utils.ModelUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Modèle pour la détection de présence.
 * Ce modèle permet de classifier si une personne est présente dans les données captées.
 */
public class PresenceModel {
    private static final Logger log = LoggerFactory.getLogger(PresenceModel.class);
    
    private final Properties config;
    private final int inputSize;
    private final int numClasses;
    private MultiLayerNetwork network;
    private ComputationGraph graphNetwork;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public PresenceModel(Properties config) {
        this.config = config;
        this.inputSize = Integer.parseInt(config.getProperty("presence.model.input.size", "100"));
        this.numClasses = Integer.parseInt(config.getProperty("presence.model.num.classes", "2"));
    }
    
    /**
     * Initialise un nouveau modèle de détection de présence.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle pour la détection de présence");
        
        // Utilisation d'un modèle simple pour la détection de présence
        network = ModelUtils.createModelFromConfig(config, "presence");
        graphNetwork = null;
    }
    
    /**
     * Charge le modèle par défaut spécifié dans la configuration.
     * S'il n'existe pas, initialise un nouveau modèle.
     *
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = config.getProperty("presence.model.path", "models/presence_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle de détection de présence par défaut depuis {}", modelPath);
            loadModel(modelPath);
        } else {
            log.warn("Modèle par défaut non trouvé à {}, initialisation d'un nouveau modèle", modelPath);
            initNewModel();
            
            // Créer le répertoire parent
            File parentDir = modelFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }
            
            // Sauvegarder le modèle nouvellement créé
            saveModel(modelPath);
        }
    }
    
    /**
     * Charge un modèle existant à partir d'un fichier.
     *
     * @param modelPath Chemin du fichier modèle
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle de détection de présence depuis {}", modelPath);
        
        try {
            network = ModelUtils.loadModel(modelPath);
            log.info("Modèle MultiLayerNetwork chargé avec succès");
            graphNetwork = null;
        } catch (Exception e) {
            log.warn("Échec du chargement comme MultiLayerNetwork, tentative de chargement comme ComputationGraph", e);
            try {
                // Tenter de charger comme ComputationGraph
                graphNetwork = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(modelPath);
                log.info("Modèle ComputationGraph chargé avec succès");
            } catch (Exception ex) {
                throw new IOException("Impossible de charger le modèle", ex);
            }
        }
    }
    
    /**
     * Sauvegarde le modèle dans un fichier.
     *
     * @param modelPath Chemin où sauvegarder le modèle
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public void saveModel(String modelPath) throws IOException {
        log.info("Sauvegarde du modèle de détection de présence vers {}", modelPath);
        
        // Créer le répertoire parent si nécessaire
        File modelFile = new File(modelPath);
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        if (network != null) {
            // Sauvegarder comme MultiLayerNetwork
            // Dans beta7, le paramètre s'appelle saveUpdater au lieu de includeUpdater
            ModelUtils.saveModel(network, modelPath, true);
            log.info("Modèle MultiLayerNetwork sauvegardé avec succès");
        } else if (graphNetwork != null) {
            // Sauvegarder comme ComputationGraph
            // Dans beta7, le paramètre s'appelle saveUpdater au lieu de includeUpdater
            org.deeplearning4j.util.ModelSerializer.writeModel(graphNetwork, modelFile, true);
            log.info("Modèle ComputationGraph sauvegardé avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle à sauvegarder");
        }
    }
    
    /**
     * Exporte le modèle au format DL4J.
     *
     * @param exportPath Chemin où exporter le modèle
     * @throws IOException Si une erreur survient lors de l'export
     */
    public void exportModel(String exportPath) throws IOException {
        log.info("Exportation du modèle de détection de présence vers {}", exportPath);
        
        int modelVersion = Integer.parseInt(config.getProperty("export.model.version", "1"));
        boolean saveUpdater = Boolean.parseBoolean(config.getProperty("export.model.include.updater", "false"));
        
        if (network != null) {
            // Exporter comme MultiLayerNetwork
            ModelUtils.exportModelForDL4J(network, exportPath, saveUpdater, modelVersion);
            log.info("Modèle MultiLayerNetwork exporté avec succès");
        } else if (graphNetwork != null) {
            // Exporter comme ComputationGraph
            File modelFile = new File(exportPath);
            org.deeplearning4j.util.ModelSerializer.writeModel(graphNetwork, modelFile, saveUpdater);
            log.info("Modèle ComputationGraph exporté avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle à exporter");
        }
    }
    
    /**
     * Obtient le réseau entraîné.
     * Note: Si le modèle utilise un ComputationGraph, null est retourné.
     *
     * @return Le réseau entraîné ou null si un ComputationGraph est utilisé
     */
    public MultiLayerNetwork getNetwork() {
        return network;
    }
    
    /**
     * Obtient le graphe de calcul entraîné.
     * Note: Si le modèle utilise un MultiLayerNetwork, null est retourné.
     *
     * @return Le graphe de calcul entraîné ou null si un MultiLayerNetwork est utilisé
     */
    public ComputationGraph getGraphNetwork() {
        return graphNetwork;
    }
    
    /**
     * Indique si le modèle est basé sur un ComputationGraph ou un MultiLayerNetwork.
     *
     * @return true si le modèle est basé sur un ComputationGraph
     */
    public boolean isGraphBased() {
        return graphNetwork != null;
    }
}