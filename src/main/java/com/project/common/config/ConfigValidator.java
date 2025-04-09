package com.project.common.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * Utilitaire de validation des configurations.
 * Permet de vérifier la cohérence et la validité des paramètres de configuration.
 */
public class ConfigValidator {
    private static final Logger log = LoggerFactory.getLogger(ConfigValidator.class);
    
    /**
     * Valide la configuration complète de l'application
     * @param config Propriétés de configuration à valider
     * @return true si la configuration est valide, false sinon
     */
    public static boolean validateConfig(Properties config) {
        boolean isValid = true;
        
        // Valider la configuration de base
        isValid &= validateBasicConfig(config);
        
        // Valider la configuration des modèles de présence
        isValid &= validatePresenceModelConfig(config);
        
        // Valider la configuration des modèles d'activité
        isValid &= validateActivityModelConfig(config);
        
        // Valider la configuration des modèles de son
        isValid &= validateSoundModelConfig(config);
        
        return isValid;
    }
    
    /**
     * Valide la configuration de base
     * @param config Propriétés de configuration
     * @return true si la configuration est valide, false sinon
     */
    private static boolean validateBasicConfig(Properties config) {
        log.info("Validation de la configuration de base...");
        boolean isValid = true;
        
        // Vérifier les chemins de base
        String[] requiredPaths = {
            "data.root.dir",
            "models.root.dir"
        };
        
        for (String path : requiredPaths) {
            if (!config.containsKey(path) || config.getProperty(path).trim().isEmpty()) {
                log.error("Configuration invalide : {} n'est pas défini", path);
                isValid = false;
            }
        }
        
        return isValid;
    }
    
    /**
     * Valide la configuration des modèles de présence
     * @param config Propriétés de configuration
     * @return true si la configuration est valide, false sinon
     */
    private static boolean validatePresenceModelConfig(Properties config) {
        log.info("Validation de la configuration des modèles de présence...");
        boolean isValid = true;
        
        // Vérifier le type de modèle
        String modelType = config.getProperty("presence.model.type", "STANDARD");
        if (!modelType.equals("STANDARD") && !modelType.equals("YOLO")) {
            log.warn("Type de modèle de présence inconnu : {}, sera traité comme STANDARD", modelType);
        }
        
        // Vérifier le nombre de classes
        try {
            int numClasses = Integer.parseInt(config.getProperty("presence.model.num.classes", "2"));
            if (numClasses < 2) {
                log.error("Nombre de classes pour le modèle de présence invalide : {}, doit être >= 2", numClasses);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            log.error("Nombre de classes pour le modèle de présence invalide : {}", 
                    config.getProperty("presence.model.num.classes"));
            isValid = false;
        }
        
        return isValid;
    }
    
    /**
     * Valide la configuration des modèles d'activité
     * @param config Propriétés de configuration
     * @return true si la configuration est valide, false sinon
     */
    private static boolean validateActivityModelConfig(Properties config) {
        log.info("Validation de la configuration des modèles d'activité...");
        boolean isValid = true;
        
        // Vérifier le type de modèle
        String modelType = config.getProperty("activity.model.type", "STANDARD");
        if (!modelType.equals("STANDARD") && !modelType.equals("VGG16") && !modelType.equals("RESNET")) {
            log.warn("Type de modèle d'activité inconnu : {}, sera traité comme STANDARD", modelType);
        }
        
        // Vérifier le nombre de classes
        try {
            int numClasses = Integer.parseInt(config.getProperty("activity.model.num.classes", "27"));
            if (numClasses < 2) {
                log.error("Nombre de classes pour le modèle d'activité invalide : {}, doit être >= 2", numClasses);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            log.error("Nombre de classes pour le modèle d'activité invalide : {}", 
                    config.getProperty("activity.model.num.classes"));
            isValid = false;
        }
        
        // Vérifier les dimensions des images
        Map<String, String> dimensionParams = new HashMap<>();
        dimensionParams.put("activity.model.input.height", "Hauteur d'image");
        dimensionParams.put("activity.model.input.width", "Largeur d'image");
        
        for (Map.Entry<String, String> entry : dimensionParams.entrySet()) {
            try {
                int dimension = Integer.parseInt(config.getProperty(entry.getKey(), "224"));
                if (dimension <= 0) {
                    log.error("{} pour le modèle d'activité invalide : {}, doit être > 0", 
                            entry.getValue(), dimension);
                    isValid = false;
                }
            } catch (NumberFormatException e) {
                log.error("{} pour le modèle d'activité invalide : {}", 
                        entry.getValue(), config.getProperty(entry.getKey()));
                isValid = false;
            }
        }
        
        return isValid;
    }
    
    /**
     * Valide la configuration des modèles de son
     * @param config Propriétés de configuration
     * @return true si la configuration est valide, false sinon
     */
    private static boolean validateSoundModelConfig(Properties config) {
        log.info("Validation de la configuration des modèles de son...");
        boolean isValid = true;
        
        // Vérifier le type de modèle
        String modelType = config.getProperty("sound.model.type", "STANDARD");
        if (!modelType.equals("STANDARD") && !modelType.equals("SPECTROGRAM")) {
            log.warn("Type de modèle de son inconnu : {}, sera traité comme STANDARD", modelType);
        }
        
        // Vérifier le nombre de classes
        try {
            int numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "8"));
            if (numClasses < 2) {
                log.error("Nombre de classes pour le modèle de son invalide : {}, doit être >= 2", numClasses);
                isValid = false;
            }
        } catch (NumberFormatException e) {
            log.error("Nombre de classes pour le modèle de son invalide : {}", 
                    config.getProperty("sound.model.num.classes"));
            isValid = false;
        }
        
        // Vérifier les paramètres du spectrogramme
        if (modelType.equals("SPECTROGRAM")) {
            Map<String, String> spectrogramParams = new HashMap<>();
            spectrogramParams.put("sound.spectrogram.height", "Hauteur du spectrogramme");
            spectrogramParams.put("sound.spectrogram.width", "Largeur du spectrogramme");
            spectrogramParams.put("sound.sample.rate", "Taux d'échantillonnage");
            spectrogramParams.put("sound.fft.size", "Taille FFT");
            
            for (Map.Entry<String, String> entry : spectrogramParams.entrySet()) {
                try {
                    int value = Integer.parseInt(config.getProperty(entry.getKey(), "0"));
                    if (value <= 0) {
                        log.error("{} pour le modèle de son invalide : {}, doit être > 0", 
                                entry.getValue(), value);
                        isValid = false;
                    }
                } catch (NumberFormatException e) {
                    log.error("{} pour le modèle de son invalide : {}", 
                            entry.getValue(), config.getProperty(entry.getKey()));
                    isValid = false;
                }
            }
        }
        
        return isValid;
    }
}
