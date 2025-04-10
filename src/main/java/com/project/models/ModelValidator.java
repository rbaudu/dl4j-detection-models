package com.project.models;

import com.project.models.activity.ActivityModel;
import com.project.models.presence.YOLOPresenceModel;
import com.project.models.sound.SoundModel;
import com.project.models.sound.SpectrogramSoundModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Classe pour valider les différents modèles de détection.
 * Permet de vérifier que les modèles peuvent être chargés correctement et sont fonctionnels.
 */
public class ModelValidator {
    private static final Logger log = LoggerFactory.getLogger(ModelValidator.class);
    
    private final Properties config;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ModelValidator(Properties config) {
        this.config = config;
    }
    
    /**
     * Valide le modèle YOLO de détection de présence.
     *
     * @return true si le modèle est valide, false sinon
     * @throws IOException si une erreur survient lors de la validation
     */
    public boolean validatePresenceModel() throws IOException {
        log.info("Validation du modèle YOLO de détection de présence");
        
        try {
            YOLOPresenceModel model = new YOLOPresenceModel(config);
            model.loadDefaultModel();
            
            // Vérifier que le modèle est bien chargé
            if (model.getYoloNetwork() != null) {
                log.info("Le modèle YOLO de détection de présence a été chargé avec succès");
                return true;
            } else {
                log.error("Le modèle YOLO de détection de présence n'a pas pu être chargé correctement");
                return false;
            }
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle YOLO de détection de présence", e);
            return false;
        }
    }
    
    /**
     * Valide le modèle de détection d'activité.
     *
     * @return true si le modèle est valide, false sinon
     * @throws IOException si une erreur survient lors de la validation
     */
    public boolean validateActivityModel() throws IOException {
        log.info("Validation du modèle de détection d'activité");
        
        try {
            ActivityModel model = new ActivityModel(config);
            model.loadDefaultModel();
            
            // Vérifier que le modèle est bien chargé
            if (model.getNetwork() != null || model.getGraphNetwork() != null) {
                log.info("Le modèle de détection d'activité a été chargé avec succès");
                return true;
            } else {
                log.error("Le modèle de détection d'activité n'a pas pu être chargé correctement");
                return false;
            }
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle de détection d'activité", e);
            return false;
        }
    }
    
    /**
     * Valide le modèle de détection de sons.
     *
     * @return true si le modèle est valide, false sinon
     * @throws IOException si une erreur survient lors de la validation
     */
    public boolean validateSoundModel() throws IOException {
        String soundModelType = config.getProperty("sound.model.type", "STANDARD");
        
        if ("SPECTROGRAM".equalsIgnoreCase(soundModelType)) {
            return validateSpectrogramSoundModel();
        } else {
            return validateStandardSoundModel();
        }
    }
    
    /**
     * Valide le modèle standard de détection de sons.
     *
     * @return true si le modèle est valide, false sinon
     * @throws IOException si une erreur survient lors de la validation
     */
    private boolean validateStandardSoundModel() throws IOException {
        log.info("Validation du modèle standard de détection de sons");
        
        try {
            SoundModel model = new SoundModel(config);
            model.loadDefaultModel();
            
            // Vérifier que le modèle est bien chargé
            if (model.getNetwork() != null || model.getGraphNetwork() != null) {
                log.info("Le modèle standard de détection de sons a été chargé avec succès");
                return true;
            } else {
                log.error("Le modèle standard de détection de sons n'a pas pu être chargé correctement");
                return false;
            }
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle standard de détection de sons", e);
            return false;
        }
    }
    
    /**
     * Valide le modèle de détection de sons basé sur spectrogrammes.
     *
     * @return true si le modèle est valide, false sinon
     * @throws IOException si une erreur survient lors de la validation
     */
    private boolean validateSpectrogramSoundModel() throws IOException {
        log.info("Validation du modèle de détection de sons basé sur spectrogrammes");
        
        try {
            SpectrogramSoundModel model = new SpectrogramSoundModel(config);
            model.loadDefaultModel();
            
            // Vérifier que le modèle est bien chargé
            if (model.getNetwork() != null) {
                log.info("Le modèle de détection de sons basé sur spectrogrammes a été chargé avec succès");
                return true;
            } else {
                log.error("Le modèle de détection de sons basé sur spectrogrammes n'a pas pu être chargé correctement");
                return false;
            }
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle de détection de sons basé sur spectrogrammes", e);
            return false;
        }
    }
    
    /**
     * Valide tous les modèles.
     *
     * @return true si tous les modèles sont valides, false sinon
     * @throws IOException si une erreur survient lors de la validation
     */
    public boolean validateAllModels() throws IOException {
        log.info("Validation de tous les modèles");
        
        boolean presenceValid = validatePresenceModel();
        boolean activityValid = validateActivityModel();
        boolean soundValid = validateSoundModel();
        
        return presenceValid && activityValid && soundValid;
    }
}
