package com.project.common.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Utilitaires pour la journalisation
 */
public class LoggingUtils {
    private static final Logger log = LoggerFactory.getLogger(LoggingUtils.class);
    private static final String SEPARATOR = "=".repeat(80);
    
    /**
     * Affiche une ligne de séparation dans les logs
     */
    public static void logSeparator() {
        log.info(SEPARATOR);
    }
    
    /**
     * Affiche un texte entouré de séparateurs
     * 
     * @param message Le message à afficher
     */
    public static void logSection(String message) {
        logSeparator();
        log.info(message);
        logSeparator();
    }
    
    /**
     * Affiche un message d'information avec un préfixe
     * 
     * @param prefix Préfixe à ajouter
     * @param message Message à afficher
     */
    public static void logInfo(String prefix, String message) {
        log.info("[{}] {}", prefix, message);
    }
    
    /**
     * Affiche un message d'erreur avec un préfixe
     * 
     * @param prefix Préfixe à ajouter
     * @param message Message à afficher
     */
    public static void logError(String prefix, String message) {
        log.error("[{}] {}", prefix, message);
    }
    
    /**
     * Affiche un message d'avertissement avec un préfixe
     * 
     * @param prefix Préfixe à ajouter
     * @param message Message à afficher
     */
    public static void logWarning(String prefix, String message) {
        log.warn("[{}] {}", prefix, message);
    }
    
    /**
     * Affiche un message d'information avec un temps d'exécution
     * 
     * @param message Message à afficher
     * @param elapsedTimeMs Temps écoulé en millisecondes
     */
    public static void logTimedInfo(String message, long elapsedTimeMs) {
        log.info("{} - Temps écoulé: {} ms", message, elapsedTimeMs);
    }
    
    /**
     * Formatte un temps en millisecondes en une chaîne lisible (HH:MM:SS.mmm)
     * 
     * @param milliseconds Temps en millisecondes
     * @return Temps formatté
     */
    public static String formatTime(long milliseconds) {
        long hours = milliseconds / (60 * 60 * 1000);
        long minutes = (milliseconds % (60 * 60 * 1000)) / (60 * 1000);
        long seconds = (milliseconds % (60 * 1000)) / 1000;
        long millis = milliseconds % 1000;
        
        return String.format("%02d:%02d:%02d.%03d", hours, minutes, seconds, millis);
    }
    
    /**
     * Log les paramètres d'entraînement pour la détection de présence
     * 
     * @param config Propriétés de configuration
     */
    public static void logPresenceTrainingParameters(Properties config) {
        logSection("Paramètres d'entraînement du modèle de détection de présence");
        
        // Log des paramètres principaux
        for (String key : config.stringPropertyNames()) {
            log.info("{} = {}", key, config.getProperty(key));
        }
        
        logSeparator();
    }
    
    /**
     * Log les paramètres d'entraînement pour la détection d'activité
     * 
     * @param config Propriétés de configuration
     */
    public static void logActivityTrainingParameters(Properties config) {
        logSection("Paramètres d'entraînement du modèle de détection d'activité");
        
        // Log des paramètres principaux
        for (String key : config.stringPropertyNames()) {
            log.info("{} = {}", key, config.getProperty(key));
        }
        
        logSeparator();
    }
    
    /**
     * Log les paramètres d'entraînement pour la détection de sons
     * 
     * @param config Propriétés de configuration
     */
    public static void logSoundTrainingParameters(Properties config) {
        logSection("Paramètres d'entraînement du modèle de détection de sons");
        
        // Log des paramètres principaux
        for (String key : config.stringPropertyNames()) {
            log.info("{} = {}", key, config.getProperty(key));
        }
        
        logSeparator();
    }
    
    /**
     * Log une exception avec un message contextuel
     * 
     * @param e Exception à logger
     * @param context Message contextuel
     */
    public static void logException(Exception e, String context) {
        log.error("Erreur: {} - {}", context, e.getMessage(), e);
    }
    
    /**
     * Log les informations sur la structure d'un modèle
     * 
     * @param modelName Nom du modèle
     * @param parameterCount Nombre de paramètres
     * @param layerSizes Tableau des tailles de couches
     * @param inputHeight Hauteur d'entrée pour les modèles CNN
     */
    public static void logModelStructureInfo(String modelName, long parameterCount, int[] layerSizes, int inputHeight) {
        logSection("Structure du modèle: " + modelName);
        
        log.info("Nombre total de paramètres: {}", parameterCount);
        
        if (layerSizes != null && layerSizes.length > 0) {
            StringBuilder sb = new StringBuilder("Tailles des couches: ");
            for (int i = 0; i < layerSizes.length; i++) {
                sb.append(layerSizes[i]);
                if (i < layerSizes.length - 1) {
                    sb.append(" -> ");
                }
            }
            log.info(sb.toString());
        }
        
        if (inputHeight > 0) {
            log.info("Hauteur d'entrée: {}", inputHeight);
        }
        
        logSeparator();
    }
}