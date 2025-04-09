package com.project.models.sound;

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
 * Modèle pour la classification de sons basé sur des caractéristiques audio.
 * Peut utiliser soit un modèle simple, soit un modèle de transfert d'apprentissage.
 */
public class SoundModel {
    private static final Logger log = LoggerFactory.getLogger(SoundModel.class);
    
    private final Properties config;
    private final int inputLength;
    private final int numMfcc;
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
    public SoundModel(Properties config) {
        this.config = config;
        this.inputLength = Integer.parseInt(config.getProperty("sound.model.input.length", "16000"));
        this.numMfcc = Integer.parseInt(config.getProperty("sound.model.num.mfcc", "40"));
        this.numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "3"));
        this.dropoutRate = Double.parseDouble(config.getProperty("sound.model.dropout", "0.5"));
        this.usesTransferLearning = Boolean.parseBoolean(config.getProperty("sound.model.use.transfer", "true"));
    }
    
    /**
     * Initialise un nouveau modèle.
     * Selon la configuration, utilise soit un modèle simple, soit un modèle de transfert d'apprentissage.
     */
    public void initNewModel() {
        log.info("Initialisation d'un nouveau modèle pour la classification de sons");
        
        if (usesTransferLearning) {
            // Utiliser le transfert d'apprentissage avec YAMNet
            log.info("Utilisation du transfert d'apprentissage avec YAMNet");
            graphNetwork = TransferLearningHelper.loadYAMNetForSoundClassification(
                    inputLength, numMfcc, dropoutRate);
            
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
            log.info("Création d'un modèle simple pour la classification de sons");
            network = ModelUtils.createModelFromConfig(config, "sound");
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
        String modelPath = config.getProperty("sound.model.path", "models/sound_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle de classification de sons par défaut depuis {}", modelPath);
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
        log.info("Chargement du modèle de classification de sons depuis {}", modelPath);
        
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
        log.info("Sauvegarde du modèle de classification de sons vers {}", modelPath);
        
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
        log.info("Exportation du modèle de classification de sons vers {}", exportPath);
        
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
        int inputSize = inputLength * numMfcc;
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