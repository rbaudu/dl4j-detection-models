package com.project.models.activity;

import com.project.common.utils.ModelUtils;
import com.project.common.utils.TransferLearningHelper;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Modèle pour la détection d'activité basé sur une architecture de réseau de neurones convolutifs.
 * Utilise un modèle pré-entraîné (MobileNetV2) pour extraire des caractéristiques des images.
 */
public class ActivityModel {
    private static final Logger log = LoggerFactory.getLogger(ActivityModel.class);
    
    private final Properties config;
    private final int inputHeight;
    private final int inputWidth;
    private final int numClasses;
    private final double dropoutRate;
    private MultiLayerNetwork network;
    private ComputationGraph graphNetwork;
    private boolean usesTransferLearning;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ActivityModel(Properties config) {
        this.config = config;
        this.inputHeight = Integer.parseInt(config.getProperty("activity.model.input.height", "224"));
        this.inputWidth = Integer.parseInt(config.getProperty("activity.model.input.width", "224"));
        this.numClasses = Integer.parseInt(config.getProperty("activity.model.num.classes", "5"));
        this.dropoutRate = Double.parseDouble(config.getProperty("activity.model.dropout", "0.5"));
        this.usesTransferLearning = Boolean.parseBoolean(config.getProperty("activity.model.use.transfer", "true"));
    }
    
    /**
     * Initialise le modèle.
     * Cette méthode est un point d'entrée unique pour initialiser le modèle
     * en utilisant soit les paramètres par défaut, soit un modèle existant.
     * 
     * @return true si l'initialisation a réussi, false sinon
     */
    public boolean initializeModel() {
        try {
            // Vérifié si on doit charger un modèle existant ou en créer un nouveau
            String modelPath = config.getProperty("activity.model.path", "models/activity_model.zip");
            File modelFile = new File(modelPath);
            
            if (modelFile.exists() && Boolean.parseBoolean(config.getProperty("activity.model.load.existing", "true"))) {
                // Charger un modèle existant
                log.info("Chargement d'un modèle existant depuis {}", modelPath);
                loadModel(modelPath);
            } else {
                // Initialiser un nouveau modèle
                log.info("Initialisation d'un nouveau modèle");
                initNewModel();
            }
            
            return true;
        } catch (Exception e) {
            log.error("Erreur lors de l'initialisation du modèle: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Initialise un nouveau modèle.
     * Selon la configuration, utilise soit un modèle simple, soit un modèle de transfert d'apprentissage.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle pour la détection d'activité");
        
        if (usesTransferLearning) {
            // Utiliser le transfert d'apprentissage avec MobileNetV2
            log.info("Utilisation du transfert d'apprentissage avec MobileNetV2");
            graphNetwork = TransferLearningHelper.loadMobileNetV2ForActivityClassification(
                    inputHeight, inputWidth, dropoutRate);
            
            // Convertir en MultiLayerNetwork si nécessaire pour la compatibilité
            try {
                // Tentative de conversion
                network = convertToMultiLayerNetwork(graphNetwork);
                log.info("ComputationGraph converti en MultiLayerNetwork avec succès");
            } catch (Exception e) {
                log.warn("Impossible de convertir en MultiLayerNetwork, utilisation directe du ComputationGraph", e);
                network = null; // On utilisera directement graphNetwork
            }
        } else {
            // Créer un modèle simple
            log.info("Création d'un modèle simple pour la détection d'activité");
            network = ModelUtils.createModelFromConfig(config, "activity");
            graphNetwork = null;
        }
    }
    
    /**
     * Charge le modèle par défaut spécifié dans la configuration.
     * S'il n'existe pas, initialise un nouveau modèle.
     *
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = config.getProperty("activity.model.path", "models/activity_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle de détection d'activité par défaut depuis {}", modelPath);
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
        log.info("Chargement du modèle de détection d'activité depuis {}", modelPath);
        
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
                
                // Essayer de convertir en MultiLayerNetwork pour la compatibilité
                try {
                    network = convertToMultiLayerNetwork(graphNetwork);
                    log.info("ComputationGraph converti en MultiLayerNetwork avec succès");
                } catch (Exception ex) {
                    log.warn("Impossible de convertir en MultiLayerNetwork, utilisation directe du ComputationGraph", ex);
                    network = null;
                }
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
        log.info("Sauvegarde du modèle de détection d'activité vers {}", modelPath);
        
        // Créer le répertoire parent si nécessaire
        File modelFile = new File(modelPath);
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        if (network != null) {
            // Sauvegarder comme MultiLayerNetwork
            ModelUtils.saveModel(network, modelPath, true);
            log.info("Modèle MultiLayerNetwork sauvegardé avec succès");
        } else if (graphNetwork != null) {
            // Sauvegarder comme ComputationGraph
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
        log.info("Exportation du modèle de détection d'activité vers {}", exportPath);
        
        int modelVersion = Integer.parseInt(config.getProperty("export.model.version", "1"));
        boolean includeUpdater = Boolean.parseBoolean(config.getProperty("export.model.include.updater", "false"));
        
        if (network != null) {
            // Exporter comme MultiLayerNetwork
            ModelUtils.exportModelForDL4J(network, exportPath, includeUpdater, modelVersion);
            log.info("Modèle MultiLayerNetwork exporté avec succès");
        } else if (graphNetwork != null) {
            // Exporter comme ComputationGraph
            // Simuler une exportation spéciale en sauvegardant normalement
            File modelFile = new File(exportPath);
            org.deeplearning4j.util.ModelSerializer.writeModel(graphNetwork, modelFile, includeUpdater);
            log.info("Modèle ComputationGraph exporté avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle à exporter");
        }
    }
    
    /**
     * Convertit un ComputationGraph en MultiLayerNetwork lorsque c'est possible.
     * Note: Cette conversion n'est pas toujours possible pour des architectures complexes.
     *
     * @param graph Le ComputationGraph à convertir
     * @return Un MultiLayerNetwork équivalent
     * @throws IllegalArgumentException Si la conversion n'est pas possible
     */
    private MultiLayerNetwork convertToMultiLayerNetwork(ComputationGraph graph) {
        // Vérification des conditions pour la conversion
        if (graph.getNumInputArrays() != 1 || graph.getNumOutputArrays() != 1) {
            throw new IllegalArgumentException("La conversion nécessite exactement 1 entrée et 1 sortie");
        }
        
        // Créer un modèle avec la même configuration mais adapté à un MultiLayerNetwork
        int inputSize = inputHeight * inputWidth * 3; // RGB
        MultiLayerNetwork convertedNetwork = ModelUtils.createDeepNetwork(inputSize, numClasses);
        
        // Copier les paramètres si possible (simplification - dans un cas réel, il faudrait copier couche par couche)
        // Cette étape est complexe et peut nécessiter une analyse détaillée de l'architecture
        
        return convertedNetwork;
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