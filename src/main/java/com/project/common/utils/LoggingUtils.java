package com.project.common.utils;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

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
     * Affiche les informations sur un modèle
     */
    public static void logModelInfo(MultiLayerNetwork model, String modelType, String architecture) {
        logSeparator();
        logger.info("=== Information sur la structure du modèle {} ({}) ===", modelType, architecture);
        logger.info("Nombre total de paramètres: {}", model.numParams());
        
        // Afficher la forme de l'entrée
        if (model.getLayerWiseConfigurations().getInputType() != null) {
            logger.info("Forme de l'entrée: {}", Arrays.toString(model.getLayerWiseConfigurations().getInputType().getShape()));
        }
        
        // Afficher le nombre de classes en sortie
        Layer outputLayer = model.getLayer(model.getnLayers() - 1);
        logger.info("Nombre de classes en sortie: {}", outputLayer.getParam("W").size(0));
        
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
}