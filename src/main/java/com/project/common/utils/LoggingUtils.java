package com.project.common.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Utilitaire pour la journalisation des paramètres et informations importantes.
 */
public class LoggingUtils {
    private static final Logger log = LoggerFactory.getLogger(LoggingUtils.class);
    
    /**
     * Journalise les paramètres de configuration importants pour l'entraînement d'un modèle d'activité.
     * 
     * @param config Propriétés de configuration
     */
    public static void logActivityTrainingParameters(Properties config) {
        log.info("=== Paramètres d'entraînement du modèle d'activité ===");
        log.info("Type de modèle: {}", config.getProperty("activity.model.type", "STANDARD"));
        log.info("Nombre de classes: {}", config.getProperty("activity.model.num.classes", "27"));
        log.info("Dimensions des images: {}x{}x{}", 
                config.getProperty("activity.model.input.height", "224"),
                config.getProperty("activity.model.input.width", "224"),
                config.getProperty("activity.image.channels", "3"));
        log.info("Taille du lot: {}", config.getProperty("activity.model.batch.size", "64"));
        log.info("Nombre d'époques: {}", config.getProperty("activity.model.epochs", "150"));
        log.info("Taux d'apprentissage: {}", config.getProperty("activity.model.learning.rate", "0.0005"));
        log.info("Répertoire de données: {}", config.getProperty("activity.data.dir", "data/activity"));
        log.info("Chemin du modèle: {}", config.getProperty("activity.model.dir", "models/activity"));
        log.info("=== Fin des paramètres d'entraînement ===");
    }
    
    /**
     * Journalise les paramètres de configuration importants pour l'entraînement d'un modèle de présence.
     * 
     * @param config Propriétés de configuration
     */
    public static void logPresenceTrainingParameters(Properties config) {
        log.info("=== Paramètres d'entraînement du modèle de présence ===");
        log.info("Type de modèle: {}", config.getProperty("presence.model.type", "STANDARD"));
        log.info("Nombre de classes: {}", config.getProperty("presence.model.num.classes", "2"));
        
        if ("YOLO".equalsIgnoreCase(config.getProperty("presence.model.type", "STANDARD"))) {
            log.info("Dimensions des images YOLO: {}x{}", 
                    config.getProperty("presence.model.input.height", "416"),
                    config.getProperty("presence.model.input.width", "416"));
            log.info("Utilisation de Tiny YOLO: {}", config.getProperty("presence.model.use.tiny.yolo", "true"));
        } else {
            log.info("Dimensions des images: {}x{}", 
                    config.getProperty("presence.model.input.height", "64"),
                    config.getProperty("presence.model.input.width", "64"));
            log.info("Nombre de couches cachées: {}", config.getProperty("presence.model.hidden.layers", "2"));
            log.info("Taille des couches cachées: {}", config.getProperty("presence.model.hidden.size", "128"));
        }
        
        log.info("Taille du lot: {}", config.getProperty("presence.model.batch.size", "32"));
        log.info("Nombre d'époques: {}", config.getProperty("presence.model.epochs", "100"));
        log.info("Taux d'apprentissage: {}", config.getProperty("presence.model.learning.rate", "0.001"));
        log.info("Répertoire de données: {}", config.getProperty("presence.data.dir", "data/presence"));
        log.info("Chemin du modèle: {}", config.getProperty("presence.model.dir", "models/presence"));
        log.info("=== Fin des paramètres d'entraînement ===");
    }
    
    /**
     * Journalise les paramètres de configuration importants pour l'entraînement d'un modèle de son.
     * 
     * @param config Propriétés de configuration
     */
    public static void logSoundTrainingParameters(Properties config) {
        log.info("=== Paramètres d'entraînement du modèle de son ===");
        log.info("Type de modèle: {}", config.getProperty("sound.model.type", "STANDARD"));
        log.info("Nombre de classes: {}", config.getProperty("sound.model.num.classes", "8"));
        
        if ("SPECTROGRAM".equalsIgnoreCase(config.getProperty("sound.model.type", "STANDARD"))) {
            log.info("Dimensions du spectrogramme: {}x{}", 
                    config.getProperty("sound.spectrogram.height", "224"),
                    config.getProperty("sound.spectrogram.width", "224"));
            log.info("Utilisation de VGG16: {}", config.getProperty("sound.model.use.vgg16", "true"));
            log.info("Taux d'échantillonnage: {}", config.getProperty("sound.sample.rate", "44100"));
            log.info("Taille FFT: {}", config.getProperty("sound.fft.size", "2048"));
            log.info("Taille de saut: {}", config.getProperty("sound.hop.size", "512"));
            log.info("Bandes Mel: {}", config.getProperty("sound.mel.bands", "128"));
        } else {
            log.info("Taille d'entrée: {}", config.getProperty("sound.model.input.size", "256"));
            log.info("Nombre de couches cachées: {}", config.getProperty("sound.model.hidden.layers", "4"));
            log.info("Taille des couches cachées: {}", config.getProperty("sound.model.hidden.size", "512"));
        }
        
        log.info("Taille du lot: {}", config.getProperty("sound.model.batch.size", "32"));
        log.info("Nombre d'époques: {}", config.getProperty("sound.model.epochs", "200"));
        log.info("Taux d'apprentissage: {}", config.getProperty("sound.model.learning.rate", "0.0001"));
        log.info("Répertoire de données: {}", config.getProperty("sound.data.dir", "data/sound"));
        log.info("Chemin du modèle: {}", config.getProperty("sound.model.dir", "models/sound"));
        log.info("=== Fin des paramètres d'entraînement ===");
    }
    
    /**
     * Journalise les informations sur la structure et les performances du modèle.
     * 
     * @param modelType Type de modèle (presence, activity, sound)
     * @param numParams Nombre de paramètres du modèle
     * @param inputShape Forme de l'entrée du modèle
     * @param outputShape Forme de la sortie du modèle
     */
    public static void logModelStructureInfo(String modelType, long numParams, int[] inputShape, int outputClasses) {
        log.info("=== Information sur la structure du modèle {} ===", modelType);
        log.info("Nombre total de paramètres: {}", numParams);
        
        StringBuilder inputShapeStr = new StringBuilder("[");
        for (int i = 0; i < inputShape.length; i++) {
            inputShapeStr.append(inputShape[i]);
            if (i < inputShape.length - 1) {
                inputShapeStr.append(", ");
            }
        }
        inputShapeStr.append("]");
        
        log.info("Forme de l'entrée: {}", inputShapeStr.toString());
        log.info("Nombre de classes en sortie: {}", outputClasses);
        log.info("=== Fin des informations sur la structure du modèle ===");
    }
    
    /**
     * Journalise les performances d'entraînement à la fin de chaque époque.
     * 
     * @param epoch Numéro de l'époque
     * @param numEpochs Nombre total d'époques
     * @param trainingAccuracy Précision sur les données d'entraînement
     * @param validationAccuracy Précision sur les données de validation
     * @param trainingLoss Perte sur les données d'entraînement
     * @param validationLoss Perte sur les données de validation
     * @param elapsedTime Temps écoulé pour cette époque (en secondes)
     */
    public static void logTrainingPerformance(int epoch, int numEpochs, double trainingAccuracy, 
                                           double validationAccuracy, double trainingLoss, 
                                           double validationLoss, long elapsedTime) {
        log.info("Époque {}/{} - Temps: {}s - Train: [précision: {:.4f}, perte: {:.4f}] - Val: [précision: {:.4f}, perte: {:.4f}]",
                epoch, numEpochs, elapsedTime, trainingAccuracy, trainingLoss, validationAccuracy, validationLoss);
    }
}
