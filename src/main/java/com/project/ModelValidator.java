// src/main/java/com/project/ModelValidator.java
package com.project;

import com.project.models.activity.ActivityModel;
import com.project.models.presence.PresenceModel;
import com.project.models.sound.SoundModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

/**
 * Classe utilitaire pour valider les modèles entraînés.
 * Cette classe effectue des vérifications de base pour s'assurer que les modèles peuvent être chargés et utilisés.
 */
public class ModelValidator {
    private static final Logger log = LoggerFactory.getLogger(ModelValidator.class);
    
    private final Properties config;
    private final Random random;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ModelValidator(Properties config) {
        this.config = config;
        this.random = new Random(Long.parseLong(config.getProperty("training.seed", "123")));
    }
    
    /**
     * Valide tous les modèles.
     *
     * @return true si tous les modèles sont valides
     */
    public boolean validateAllModels() {
        log.info("Validation de tous les modèles");
        boolean presenceValid = validatePresenceModel();
        boolean activityValid = validateActivityModel();
        boolean soundValid = validateSoundModel();
        
        boolean allValid = presenceValid && activityValid && soundValid;
        if (allValid) {
            log.info("Tous les modèles sont valides et peuvent être exportés");
        } else {
            log.error("Certains modèles n'ont pas passé la validation");
        }
        
        return allValid;
    }
    
    /**
     * Valide le modèle de détection de présence.
     *
     * @return true si le modèle est valide
     */
    public boolean validatePresenceModel() {
        log.info("Validation du modèle de détection de présence");
        String modelDir = config.getProperty("presence.model.dir", "models/presence");
        String modelName = config.getProperty("presence.model.name", "presence_model");
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        int inputSize = Integer.parseInt(config.getProperty("presence.model.input.size", "64"));
        
        try {
            // Essayer de charger le modèle
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                log.error("Le modèle n'existe pas: {}", modelPath);
                return false;
            }
            
            MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
            log.info("Modèle chargé avec succès");
            
            // Essayer de faire une prédiction
            INDArray input = Nd4j.rand(new int[]{1, inputSize});
            INDArray output = model.output(input);
            log.info("Prédiction effectuée avec succès, forme de sortie: {}", output.shape());
            
            return true;
        } catch (Exception e) {
            log.error("Échec de la validation du modèle de présence", e);
            return false;
        }
    }
    
    /**
     * Valide le modèle de détection d'activité.
     *
     * @return true si le modèle est valide
     */
    public boolean validateActivityModel() {
        log.info("Validation du modèle de détection d'activité");
        String modelDir = config.getProperty("activity.model.dir", "models/activity");
        String modelName = config.getProperty("activity.model.name", "activity_model");
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        int inputSize = Integer.parseInt(config.getProperty("activity.model.input.size", "128"));
        
        try {
            // Essayer de charger le modèle
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                log.error("Le modèle n'existe pas: {}", modelPath);
                return false;
            }
            
            MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
            log.info("Modèle chargé avec succès");
            
            // Essayer de faire une prédiction
            INDArray input = Nd4j.rand(new int[]{1, inputSize});
            INDArray output = model.output(input);
            log.info("Prédiction effectuée avec succès, forme de sortie: {}", output.shape());
            
            return true;
        } catch (Exception e) {
            log.error("Échec de la validation du modèle d'activité", e);
            return false;
        }
    }
    
    /**
     * Valide le modèle de détection de sons.
     *
     * @return true si le modèle est valide
     */
    public boolean validateSoundModel() {
        log.info("Validation du modèle de détection de sons");
        String modelDir = config.getProperty("sound.model.dir", "models/sound");
        String modelName = config.getProperty("sound.model.name", "sound_model");
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        int inputSize = Integer.parseInt(config.getProperty("sound.model.input.size", "256"));
        
        try {
            // Essayer de charger le modèle
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                log.error("Le modèle n'existe pas: {}", modelPath);
                return false;
            }
            
            MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
            log.info("Modèle chargé avec succès");
            
            // Essayer de faire une prédiction
            INDArray input = Nd4j.rand(new int[]{1, inputSize});
            INDArray output = model.output(input);
            log.info("Prédiction effectuée avec succès, forme de sortie: {}", output.shape());
            
            return true;
        } catch (Exception e) {
            log.error("Échec de la validation du modèle de sons", e);
            return false;
        }
    }
}