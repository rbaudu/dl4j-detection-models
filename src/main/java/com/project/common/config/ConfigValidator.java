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
        
        // Vérifier que la hauteur d'image est valide (>0)
        try {
            String heightStr = config.getProperty("activity.model.image.height", "0");
            int height = Integer.parseInt(heightStr);
            if (height <= 0) {
                logger.error("Hauteur d'image pour le modèle d'activité invalide : {}, doit être > 0", height);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            logger.error("La hauteur d'image pour le modèle d'activité n'est pas un nombre valide");
            isValid = false;
        }
        
        // Vérifier le type de modèle d'activité
        String modelType = config.getProperty("activity.model.type", "STANDARD");
        if (!isValidActivityModelType(modelType)) {
            logger.warn("Type de modèle d'activité inconnu : {}, sera traité comme STANDARD", modelType);
        }
        
        return isValid;
    }
    
    /**
     * Vérifie si le type de modèle d'activité est valide
     */
    private boolean isValidActivityModelType(String modelType) {
        return "STANDARD".equals(modelType) || 
               "TRANSFER_LEARNING".equals(modelType) || 
               "TINY_YOLO".equals(modelType);
    }
    
    /**
     * Valide la configuration des modèles de son
     */
    public boolean validateSoundModelConfig(Properties config) {
        logger.info("Validation de la configuration des modèles de son...");
        // Ajouter validation spécifique si nécessaire
        return true;
    }
}