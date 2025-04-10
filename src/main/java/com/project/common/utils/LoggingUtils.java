package com.project.common.utils;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Utilitaires pour la journalisation
 */
public class LoggingUtils {
    private static final Logger logger = LoggerFactory.getLogger(LoggingUtils.class);
    
    /**
     * Affiche un séparateur dans les logs
     */
    public static void logSeparator() {
        logger.info("-------------------------------------------------------");
    }
    
    /**
     * Affiche les informations sur un modèle (version adaptée aux APIs disponibles)
     */
    public static void logModelInfo(MultiLayerNetwork model, String modelType, String architecture) {
        logSeparator();
        logger.info("=== Information sur la structure du modèle {} ({}) ===", modelType, architecture);
        logger.info("Nombre total de paramètres: {}", model.numParams());
        
        // Adaptation pour l'accès à la forme d'entrée (sans InputType)
        if (model.getLayerWiseConfigurations() != null) {
            logger.info("Configuration du modèle: {}", model.getLayerWiseConfigurations().toString());
        }
        
        // Afficher le nombre de classes en sortie
        Layer outputLayer = model.getLayer(model.getnLayers() - 1);
        if (outputLayer != null && outputLayer.getParam("W") != null) {
            logger.info("Nombre de classes en sortie: {}", outputLayer.getParam("W").size(0));
        }
        
        // Afficher les formes des couches
        for (int i = 0; i < model.getnLayers(); i++) {
            Layer layer = model.getLayer(i);
            logger.debug("Couche {}: {}, Paramètres: {}", i, layer.type(), layer.numParams());
        }
        
        logger.info("=== Fin des informations sur la structure du modèle ===");
        logSeparator();
    }
    
    /**
     * Affiche les informations de progression d'entraînement
     */
    public static void logTrainingProgress(int epoch, int totalEpochs, double accuracy, double loss, long timeMs) {
        logger.info("Époque {}/{} - Accuracy: {}, Loss: {}, Temps: {} ms", 
                  epoch, totalEpochs, String.format("%.4f", accuracy), 
                  String.format("%.4f", loss), timeMs);
    }
    
    /**
     * Affiche les informations de fin d'entraînement
     */
    public static void logTrainingComplete(String modelPath, double accuracy, double precision, 
                                         double recall, double f1, long totalTimeMs) {
        logSeparator();
        logger.info("=== Entraînement terminé ===");
        logger.info("Modèle sauvegardé: {}", modelPath);
        logger.info("Métriques finales:");
        logger.info("  - Accuracy: {}", String.format("%.4f", accuracy));
        logger.info("  - Precision: {}", String.format("%.4f", precision));
        logger.info("  - Recall: {}", String.format("%.4f", recall));
        logger.info("  - F1-Score: {}", String.format("%.4f", f1));
        logger.info("Temps total d'entraînement: {} ms", totalTimeMs);
        logger.info("=== Fin du rapport d'entraînement ===");
        logSeparator();
    }
    
    /**
     * Affiche les détails d'une exception
     */
    public static void logException(Exception e, String context) {
        logger.error("Exception dans {}: {}", context, e.getMessage());
        for (StackTraceElement element : e.getStackTrace()) {
            if (element.getClassName().startsWith("com.project")) {
                logger.error("  at {}.{}({}:{})", 
                           element.getClassName(), element.getMethodName(), 
                           element.getFileName(), element.getLineNumber());
            }
        }
    }
    
    /**
     * Affiche les paramètres d'entraînement pour les modèles de présence
     */
    public static void logPresenceTrainingParameters(Properties config) {
        logSeparator();
        logger.info("=== Paramètres d'entraînement du modèle de présence ===");
        logger.info("Type de modèle: {}", config.getProperty("presence.model.type", "STANDARD"));
        logger.info("Nombre de classes: {}", config.getProperty("presence.model.num.classes", "2"));
        logger.info("Taille du lot: {}", config.getProperty("training.batch.size", "32"));
        logger.info("Nombre d'époques: {}", config.getProperty("training.epochs", "100"));
        logger.info("Taux d'apprentissage: {}", config.getProperty("training.learning.rate", "0.0001"));
        logger.info("=== Fin des paramètres d'entraînement ===");
        logSeparator();
    }
    
    /**
     * Affiche les paramètres d'entraînement pour les modèles d'activité
     */
    public static void logActivityTrainingParameters(Properties config) {
        logSeparator();
        logger.info("=== Paramètres d'entraînement du modèle d'activité ===");
        logger.info("Type de modèle: {}", config.getProperty("activity.model.type", "STANDARD"));
        logger.info("Nombre de classes: {}", config.getProperty("activity.model.num.classes", "5"));
        logger.info("Dimensions d'image: {}x{}", 
                  config.getProperty("activity.model.image.width", "224"),
                  config.getProperty("activity.model.image.height", "224"));
        logger.info("Taille du lot: {}", config.getProperty("training.batch.size", "32"));
        logger.info("Nombre d'époques: {}", config.getProperty("training.epochs", "100"));
        logger.info("Taux d'apprentissage: {}", config.getProperty("training.learning.rate", "0.0001"));
        logger.info("=== Fin des paramètres d'entraînement ===");
        logSeparator();
    }
    
    /**
     * Affiche les paramètres d'entraînement pour les modèles de son
     */
    public static void logSoundTrainingParameters(Properties config) {
        logSeparator();
        logger.info("=== Paramètres d'entraînement du modèle de son ===");
        logger.info("Type de modèle: {}", config.getProperty("sound.model.type", "STANDARD"));
        logger.info("Nombre de classes: {}", config.getProperty("sound.model.num.classes", "5"));
        logger.info("Longueur d'entrée: {}", config.getProperty("sound.input.length", "16000"));
        logger.info("Nombre de MFCC: {}", config.getProperty("sound.num.mfcc", "40"));
        logger.info("Taille du lot: {}", config.getProperty("training.batch.size", "32"));
        logger.info("Nombre d'époques: {}", config.getProperty("training.epochs", "100"));
        logger.info("Taux d'apprentissage: {}", config.getProperty("training.learning.rate", "0.0001"));
        logger.info("=== Fin des paramètres d'entraînement ===");
        logSeparator();
    }
}