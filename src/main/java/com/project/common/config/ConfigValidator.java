package com.project.common.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class ConfigValidator {
    private static final Logger logger = LoggerFactory.getLogger(ConfigValidator.class);
    
    /**
     * Valide la configuration complète
     * 
     * @param config Configuration à valider
     * @return true si la configuration est valide, false sinon
     */
    public boolean validateConfig(Properties config) {
        boolean isValid = true;
        
        isValid &= validateBaseConfig(config);
        isValid &= validatePresenceModelConfig(config);
        isValid &= validateActivityModelConfig(config);
        isValid &= validateSoundModelConfig(config);
        
        return isValid;
    }
    
    /**
     * Fonction statique publique pour faciliter la validation sans instanciation
     * 
     * @param config Configuration à valider 
     * @return true si la configuration est valide, false sinon
     */
    public static boolean validateStaticConfig(Properties config) {
        ConfigValidator validator = new ConfigValidator();
        return validator.validateConfig(config);
    }
    
    /**
     * Valide la configuration de base
     */
    public boolean validateBaseConfig(Properties config) {
        logger.info("Validation de la configuration de base...");
        boolean isValid = true;
        
        // Vérification des répertoires obligatoires
        if (config.getProperty("data.root.dir") == null) {
            logger.error("Configuration invalide : data.root.dir n'est pas défini");
            isValid = false;
        }
        
        if (config.getProperty("models.root.dir") == null) {
            logger.error("Configuration invalide : models.root.dir n'est pas défini");
            isValid = false;
        }
        
        return isValid;
    }
    
    /**
     * Valide la configuration des modèles de présence
     */
    public boolean validatePresenceModelConfig(Properties config) {
        logger.info("Validation de la configuration des modèles de présence...");
        // Ajouter validation spécifique si nécessaire
        return true;
    }
    
    /**
     * Valide la configuration des modèles d'activité
     */
    public boolean validateActivityModelConfig(Properties config) {
        logger.info("Validation de la configuration des modèles d'activité...");
        boolean isValid = true;
        
        // Vérifier que le nombre de classes est valide (>=2)
        try {
            String numClassesStr = config.getProperty("activity.model.num.classes", "0");
            int numClasses = Integer.parseInt(numClassesStr);
            if (numClasses < 2) {
                logger.error("Nombre de classes pour le modèle d'activité invalide : {}, doit être >= 2", numClasses);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            logger.error("Le nombre de classes pour le modèle d'activité n'est pas un nombre valide");
            isValid = false;
        }
        
        // Vérifier les dimensions de l'image (hauteur)
        // Essayer d'abord activity.model.input.height puis activity.model.image.height
        String heightStr = config.getProperty("activity.model.input.height");
        if (heightStr == null) {
            heightStr = config.getProperty("activity.model.image.height", "0");
        }
        
        try {
            int height = Integer.parseInt(heightStr);
            if (height <= 0) {
                logger.error("Hauteur d'image pour le modèle d'activité invalide : {}, doit être > 0", height);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            logger.error("La hauteur d'image pour le modèle d'activité n'est pas un nombre valide");
            isValid = false;
        }
        
        // Vérifier les dimensions de l'image (largeur)
        // Essayer d'abord activity.model.input.width puis activity.model.image.width
        String widthStr = config.getProperty("activity.model.input.width");
        if (widthStr == null) {
            widthStr = config.getProperty("activity.model.image.width", "0");
        }
        
        try {
            int width = Integer.parseInt(widthStr);
            if (width <= 0) {
                logger.error("Largeur d'image pour le modèle d'activité invalide : {}, doit être > 0", width);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            logger.error("La largeur d'image pour le modèle d'activité n'est pas un nombre valide");
            isValid = false;
        }
        
        // Vérifier le type de modèle d'activité
        String modelType = config.getProperty("activity.model.type", "STANDARD");
        if (!isValidActivityModelType(modelType)) {
            logger.warn("Type de modèle d'activité inconnu : {}, sera traité comme STANDARD", modelType);
            // Ne pas échouer la validation pour un type de modèle inconnu, juste un avertissement
        }
        
        return isValid;
    }
    
    /**
     * Vérifie si le type de modèle d'activité est valide
     */
    private boolean isValidActivityModelType(String modelType) {
        return "STANDARD".equals(modelType) || 
               "TRANSFER_LEARNING".equals(modelType) || 
               "TINY_YOLO".equals(modelType) ||
               "VGG16".equals(modelType) ||
               "YOLO".equals(modelType);
    }
    
    /**
     * Valide la configuration des modèles de son
     */
    public boolean validateSoundModelConfig(Properties config) {
        logger.info("Validation de la configuration des modèles de son...");
        boolean isValid = true;
        
        // Vérifier que le nombre de classes est valide (>=2)
        try {
            String numClassesStr = config.getProperty("sound.model.num.classes", "0");
            int numClasses = Integer.parseInt(numClassesStr);
            if (numClasses < 2) {
                logger.error("Nombre de classes pour le modèle de son invalide : {}, doit être >= 2", numClasses);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            logger.error("Le nombre de classes pour le modèle de son n'est pas un nombre valide");
            isValid = false;
        }
        
        // Vérifier la hauteur du spectrogramme
        // Essayer d'abord sound.spectrogram.height puis sound.model.spectrogram.height
        String spectroHeightStr = config.getProperty("sound.spectrogram.height");
        if (spectroHeightStr == null) {
            spectroHeightStr = config.getProperty("sound.model.spectrogram.height", "0");
        }
        
        try {
            int spectroHeight = Integer.parseInt(spectroHeightStr);
            if (spectroHeight <= 0) {
                logger.warn("Hauteur du spectrogramme invalide : {}, doit être > 0", spectroHeight);
                // Juste un avertissement, pas une erreur invalidant la configuration
            }
        } catch (NumberFormatException e) {
            logger.warn("La hauteur du spectrogramme n'est pas un nombre valide");
            // Juste un avertissement, pas une erreur invalidant la configuration
        }
        
        // Vérifier les types de modèle et d'entraîneur de son
        String modelType = config.getProperty("sound.model.type", "STANDARD");
        if (!isValidSoundModelType(modelType)) {
            logger.warn("Type de modèle de son inconnu : {}, sera traité comme STANDARD", modelType);
            // Ne pas échouer la validation pour un type de modèle inconnu, juste un avertissement
        }
        
        String trainerType = config.getProperty("sound.trainer.type", "MFCC");
        if (!isValidSoundTrainerType(trainerType)) {
            logger.warn("Type d'entraîneur de son inconnu : {}, sera traité comme MFCC", trainerType);
            // Ne pas échouer la validation pour un type d'entraîneur inconnu, juste un avertissement
        }
        
        return isValid;
    }
    
    /**
     * Vérifie si le type de modèle de son est valide
     */
    private boolean isValidSoundModelType(String modelType) {
        return "STANDARD".equals(modelType) || 
               "SPECTROGRAM".equals(modelType) || 
               "MFCC".equals(modelType);
    }
    
    /**
     * Vérifie si le type d'entraîneur de son est valide
     */
    private boolean isValidSoundTrainerType(String trainerType) {
        return "MFCC".equals(trainerType) || 
               "SPECTROGRAM".equals(trainerType) || 
               "SPECTROGRAM_VGG16".equals(trainerType);
    }
}