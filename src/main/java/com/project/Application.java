package com.project;

import com.project.common.config.ConfigValidator;
import com.project.common.utils.LoggingUtils;
import com.project.training.SoundTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Properties;

/**
 * Classe principale de l'application
 */
public class Application {
    private static final Logger logger = LoggerFactory.getLogger(Application.class);
    private static final String CONFIG_PATH = "config/application.properties";
    
    public static void main(String[] args) {
        logger.info("Démarrage de l'application de modèles de détection");
        
        // Charger la configuration
        Properties config = loadConfig(args.length > 0 ? args[0] : CONFIG_PATH);
        if (config == null) {
            logger.error("Impossible de charger la configuration. Arrêt de l'application.");
            return;
        }
        
        // Valider la configuration en utilisant la méthode statique
        if (!ConfigValidator.validateStaticConfig(config)) {
            logger.error("La configuration est invalide. Arrêt de l'application.");
            return;
        }
        
        // Déterminer le mode d'exécution
        String mode = args.length > 1 ? args[1] : "train";
        
        switch (mode.toLowerCase()) {
            case "train":
                trainModels(config, args);
                break;
            case "evaluate":
                evaluateModels(config, args);
                break;
            case "predict":
                predictWithModels(config, args);
                break;
            default:
                logger.error("Mode non reconnu: {}. Utiliser 'train', 'evaluate' ou 'predict'.", mode);
        }
        
        logger.info("Application terminée");
    }
    
    /**
     * Charge la configuration depuis un fichier
     */
    private static Properties loadConfig(String configPath) {
        Properties config = new Properties();
        
        try (FileInputStream input = new FileInputStream(configPath)) {
            config.load(input);
            logger.info("Configuration chargée depuis: {}", configPath);
            return config;
        } catch (IOException e) {
            logger.error("Erreur lors du chargement de la configuration: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Entraîne les modèles selon les paramètres
     */
    private static void trainModels(Properties config, String[] args) {
        String modelType = args.length > 2 ? args[2] : "all";
        
        switch (modelType.toLowerCase()) {
            case "presence":
                trainPresenceModel(config);
                break;
            case "activity":
                trainActivityModel(config);
                break;
            case "sound":
                trainSoundModel(config);
                break;
            case "all":
                trainPresenceModel(config);
                trainActivityModel(config);
                trainSoundModel(config);
                break;
            default:
                logger.error("Type de modèle non reconnu: {}. Utiliser 'presence', 'activity', 'sound' ou 'all'.", modelType);
        }
    }
    
    /**
     * Entraîne un modèle de présence
     */
    private static void trainPresenceModel(Properties config) {
        logger.info("Entraînement du modèle de présence");
        LoggingUtils.logPresenceTrainingParameters(config);
        
        // TODO: Implémenter l'entraînement du modèle de présence
        
        logger.info("Entraînement du modèle de présence terminé");
    }
    
    /**
     * Entraîne un modèle d'activité
     */
    private static void trainActivityModel(Properties config) {
        logger.info("Entraînement du modèle d'activité");
        LoggingUtils.logActivityTrainingParameters(config);
        
        // TODO: Implémenter l'entraînement du modèle d'activité
        
        logger.info("Entraînement du modèle d'activité terminé");
    }
    
    /**
     * Entraîne un modèle de son
     */
    private static void trainSoundModel(Properties config) {
        logger.info("Entraînement du modèle de son");
        LoggingUtils.logSoundTrainingParameters(config);
        
        // Utiliser la factory pour créer le bon type d'entraîneur
        String soundTrainerType = config.getProperty("sound.trainer.type", "MFCC");
        SoundTrainer trainer = SoundTrainer.createTrainer(soundTrainerType, config);
        
        // Entraîner le modèle
        try {
            trainer.train();
            logger.info("Entraînement du modèle de son terminé");
        } catch (Exception e) {
            logger.error("Erreur lors de l'entraînement du modèle de son: {}", e.getMessage());
            LoggingUtils.logException(e, "trainSoundModel");
        }
    }
    
    /**
     * Évalue les modèles selon les paramètres
     */
    private static void evaluateModels(Properties config, String[] args) {
        String modelType = args.length > 2 ? args[2] : "all";
        
        switch (modelType.toLowerCase()) {
            case "presence":
                evaluatePresenceModel(config);
                break;
            case "activity":
                evaluateActivityModel(config);
                break;
            case "sound":
                evaluateSoundModel(config);
                break;
            case "all":
                evaluatePresenceModel(config);
                evaluateActivityModel(config);
                evaluateSoundModel(config);
                break;
            default:
                logger.error("Type de modèle non reconnu: {}. Utiliser 'presence', 'activity', 'sound' ou 'all'.", modelType);
        }
    }
    
    /**
     * Évalue un modèle de présence
     */
    private static void evaluatePresenceModel(Properties config) {
        logger.info("Évaluation du modèle de présence");
        LoggingUtils.logPresenceTrainingParameters(config);
        
        // TODO: Implémenter l'évaluation du modèle de présence
        
        logger.info("Évaluation du modèle de présence terminée");
    }
    
    /**
     * Évalue un modèle d'activité
     */
    private static void evaluateActivityModel(Properties config) {
        logger.info("Évaluation du modèle d'activité");
        LoggingUtils.logActivityTrainingParameters(config);
        
        // TODO: Implémenter l'évaluation du modèle d'activité
        
        logger.info("Évaluation du modèle d'activité terminée");
    }
    
    /**
     * Évalue un modèle de son
     */
    private static void evaluateSoundModel(Properties config) {
        logger.info("Évaluation du modèle de son");
        LoggingUtils.logSoundTrainingParameters(config);
        
        // Utiliser la factory pour créer le bon type d'entraîneur
        String soundTrainerType = config.getProperty("sound.trainer.type", "MFCC");
        try {
            SoundTrainer trainer = SoundTrainer.createTrainer(soundTrainerType, config);
            // TODO: Implémenter l'évaluation du modèle de son
            
            logger.info("Évaluation du modèle de son terminée");
        } catch (Exception e) {
            logger.error("Erreur lors de l'évaluation du modèle de son: {}", e.getMessage());
            LoggingUtils.logException(e, "evaluateSoundModel");
        }
    }
    
    /**
     * Fait des prédictions avec les modèles
     */
    private static void predictWithModels(Properties config, String[] args) {
        String modelType = args.length > 2 ? args[2] : "all";
        
        switch (modelType.toLowerCase()) {
            case "presence":
                predictWithPresenceModel(config, args);
                break;
            case "activity":
                predictWithActivityModel(config, args);
                break;
            case "sound":
                predictWithSoundModel(config, args);
                break;
            case "all":
                predictWithPresenceModel(config, args);
                predictWithActivityModel(config, args);
                predictWithSoundModel(config, args);
                break;
            default:
                logger.error("Type de modèle non reconnu: {}. Utiliser 'presence', 'activity', 'sound' ou 'all'.", modelType);
        }
    }
    
    /**
     * Fait une prédiction avec un modèle de présence
     */
    private static void predictWithPresenceModel(Properties config, String[] args) {
        logger.info("Prédiction avec le modèle de présence");
        
        if (args.length <= 3) {
            logger.error("Chemin d'entrée manquant pour la prédiction de présence");
            return;
        }
        
        String inputPath = args[3];
        if (!Files.exists(Paths.get(inputPath))) {
            logger.error("Fichier d'entrée introuvable: {}", inputPath);
            return;
        }
        
        // TODO: Implémenter la prédiction avec le modèle de présence
        
        logger.info("Prédiction avec le modèle de présence terminée");
    }
    
    /**
     * Fait une prédiction avec un modèle d'activité
     */
    private static void predictWithActivityModel(Properties config, String[] args) {
        logger.info("Prédiction avec le modèle d'activité");
        
        if (args.length <= 3) {
            logger.error("Chemin d'entrée manquant pour la prédiction d'activité");
            return;
        }
        
        String inputPath = args[3];
        if (!Files.exists(Paths.get(inputPath))) {
            logger.error("Fichier d'entrée introuvable: {}", inputPath);
            return;
        }
        
        // TODO: Implémenter la prédiction avec le modèle d'activité
        
        logger.info("Prédiction avec le modèle d'activité terminée");
    }
    
    /**
     * Fait une prédiction avec un modèle de son
     */
    private static void predictWithSoundModel(Properties config, String[] args) {
        logger.info("Prédiction avec le modèle de son");
        
        if (args.length <= 3) {
            logger.error("Chemin d'entrée manquant pour la prédiction de son");
            return;
        }
        
        String inputPath = args[3];
        if (!Files.exists(Paths.get(inputPath))) {
            logger.error("Fichier d'entrée introuvable: {}", inputPath);
            return;
        }
        
        // Utiliser la factory pour créer le bon type d'entraîneur
        String soundTrainerType = config.getProperty("sound.trainer.type", "MFCC");
        try {
            SoundTrainer trainer = SoundTrainer.createTrainer(soundTrainerType, config);
            // TODO: Implémenter la prédiction avec le modèle de son
            
            logger.info("Prédiction avec le modèle de son terminée");
        } catch (Exception e) {
            logger.error("Erreur lors de la prédiction avec le modèle de son: {}", e.getMessage());
            LoggingUtils.logException(e, "predictWithSoundModel");
        }
    }
}