package com.project.models.presence;

import com.project.common.utils.ModelUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Modèle YOLO pour la détection de présence.
 * Utilise un modèle YOLO pré-entraîné depuis DL4J-zoo pour la détection d'objets et de personnes.
 */
public class YOLOPresenceModel {
    private static final Logger log = LoggerFactory.getLogger(YOLOPresenceModel.class);
    
    private final Properties config;
    private final int inputHeight;
    private final int inputWidth;
    private final int channels;
    private final boolean useTinyYOLO; // Utiliser TinyYOLO (plus léger) ou YOLO2 (plus précis)
    private ComputationGraph yoloNetwork;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public YOLOPresenceModel(Properties config) {
        this.config = config;
        this.inputHeight = Integer.parseInt(config.getProperty("presence.model.input.height", "416"));
        this.inputWidth = Integer.parseInt(config.getProperty("presence.model.input.width", "416"));
        this.channels = 3; // RGB
        this.useTinyYOLO = Boolean.parseBoolean(config.getProperty("presence.model.use.tiny.yolo", "true"));
    }
    
    /**
     * Initialise un nouveau modèle YOLO pour la détection de présence.
     * Charge soit TinyYOLO (plus rapide) soit YOLO2 (plus précis) selon la configuration.
     * 
     * @throws IOException Si une erreur survient lors du chargement des poids pré-entraînés
     */
    public void initNewModel() throws IOException {
        log.info("Initialisation d'un nouveau modèle YOLO pour la détection de présence");
        
        ZooModel<?> zooModel;
        if (useTinyYOLO) {
            log.info("Utilisation du modèle TinyYOLO (plus léger)");
            zooModel = TinyYOLO.builder()
                    .numClasses(20)  // Classes COCO/VOC par défaut
                    .inputShape(new int[]{channels, inputHeight, inputWidth})
                    .build();
        } else {
            log.info("Utilisation du modèle YOLO2 (plus précis)");
            zooModel = YOLO2.builder()
                    .numClasses(20)  // Classes COCO/VOC par défaut
                    .inputShape(new int[]{channels, inputHeight, inputWidth})
                    .build();
        }
        
        // Charger les poids pré-entraînés
        yoloNetwork = (ComputationGraph) zooModel.initPretrained();
        log.info("Modèle YOLO chargé avec succès");
    }
    
    /**
     * Charge le modèle par défaut spécifié dans la configuration.
     * S'il n'existe pas, initialise un nouveau modèle.
     *
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = config.getProperty("presence.yolo.model.path", "models/presence/yolo_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle YOLO de détection de présence depuis {}", modelPath);
            loadModel(modelPath);
        } else {
            log.warn("Modèle YOLO par défaut non trouvé à {}, initialisation d'un nouveau modèle", modelPath);
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
     * Charge un modèle YOLO existant à partir d'un fichier.
     *
     * @param modelPath Chemin du fichier modèle
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle YOLO de détection de présence depuis {}", modelPath);
        
        try {
            // Les modèles YOLO sont des ComputationGraph
            yoloNetwork = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(modelPath);
            log.info("Modèle YOLO chargé avec succès");
        } catch (Exception e) {
            throw new IOException("Impossible de charger le modèle YOLO", e);
        }
    }
    
    /**
     * Sauvegarde le modèle YOLO dans un fichier.
     *
     * @param modelPath Chemin où sauvegarder le modèle
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public void saveModel(String modelPath) throws IOException {
        log.info("Sauvegarde du modèle YOLO de détection de présence vers {}", modelPath);
        
        // Créer le répertoire parent si nécessaire
        File modelFile = new File(modelPath);
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        if (yoloNetwork != null) {
            // Sauvegarder comme ComputationGraph
            org.deeplearning4j.util.ModelSerializer.writeModel(yoloNetwork, modelFile, true);
            log.info("Modèle YOLO sauvegardé avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle YOLO à sauvegarder");
        }
    }
    
    /**
     * Exporte le modèle YOLO au format DL4J.
     *
     * @param exportPath Chemin où exporter le modèle
     * @throws IOException Si une erreur survient lors de l'export
     */
    public void exportModel(String exportPath) throws IOException {
        log.info("Exportation du modèle YOLO de détection de présence vers {}", exportPath);
        
        boolean includeUpdater = Boolean.parseBoolean(config.getProperty("export.model.include.updater", "false"));
        
        if (yoloNetwork != null) {
            // Exporter comme ComputationGraph
            File modelFile = new File(exportPath);
            org.deeplearning4j.util.ModelSerializer.writeModel(yoloNetwork, modelFile, includeUpdater);
            log.info("Modèle YOLO exporté avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle YOLO à exporter");
        }
    }
    
    /**
     * Détecte la présence d'une personne dans une image.
     * 
     * @param imageData Image sous forme de tableau de pixels RGB (hauteur x largeur x 3)
     * @return true si une personne est détectée, false sinon
     */
    public boolean detectPresence(INDArray imageData) {
        if (yoloNetwork == null) {
            throw new IllegalStateException("Le modèle YOLO n'est pas initialisé");
        }
        
        // Prétraitement de l'image
        INDArray input = preprocessImage(imageData);
        
        // Faire la prédiction avec YOLO
        INDArray output = yoloNetwork.outputSingle(input);
        
        // Analyser les résultats pour détecter les personnes (classe d'index 14 dans COCO/VOC)
        // Dans un cas réel, il faudrait implémenter la logique de décodage des boîtes YOLO
        boolean personDetected = analyzeYOLOOutput(output);
        
        return personDetected;
    }
    
    /**
     * Prétraitement de l'image pour la rendre compatible avec YOLO.
     * 
     * @param imageData Image d'entrée
     * @return Image prétraitée
     */
    private INDArray preprocessImage(INDArray imageData) {
        // Redimensionner l'image si nécessaire
        INDArray resizedImage = imageData; // Supposons que l'image est déjà à la bonne taille
        
        // Normaliser l'image (0-255 -> 0-1)
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(resizedImage);
        
        return resizedImage;
    }
    
    /**
     * Analyse la sortie YOLO pour déterminer si une personne est présente.
     * Dans une implémentation réelle, cela impliquerait de décoder les boîtes YOLO
     * et de filtrer celles correspondant à des personnes.
     * 
     * @param yoloOutput Sortie du réseau YOLO
     * @return true si une personne est détectée, false sinon
     */
    private boolean analyzeYOLOOutput(INDArray yoloOutput) {
        // Logique simplifiée pour la démonstration
        // Dans un cas réel, il faudrait implémenter la logique de décodage des boîtes YOLO
        
        // Supposons que nous avons décodé les boîtes et vérifié si la classe "personne" est présente
        boolean personDetected = false;
        
        // Logique simplifiée: si la confiance pour la classe "personne" (index 14) est > 0.5 
        // Note: ce n'est qu'une simplification, le format réel de sortie de YOLO est plus complexe
        
        // TODO: Implémenter la vraie logique de décodage YOLO
        
        return personDetected;
    }
    
    /**
     * Obtient le réseau YOLO.
     *
     * @return Le réseau YOLO
     */
    public ComputationGraph getYoloNetwork() {
        return yoloNetwork;
    }
}
