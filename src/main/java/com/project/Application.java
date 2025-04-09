package com.project;

import com.project.common.config.ConfigLoader;
import com.project.common.utils.DataProcessor;
import com.project.export.ActivityExporter;
import com.project.export.PresenceExporter;
import com.project.export.SoundExporter;
import com.project.models.ModelValidator;
import com.project.training.ActivityTrainer;
import com.project.training.PresenceTrainer;
import com.project.training.SoundTrainer;
import com.project.training.YOLOPresenceTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Point d'entrée principal de l'application.
 * Cette classe gère l'exécution des différents processus d'entraînement et d'export des modèles.
 */
public class Application {
    private static final Logger log = LoggerFactory.getLogger(Application.class);

    public static void main(String[] args) {
        log.info("Démarrage de l'application de modèles de détection avec DL4J");
        
        try {
            // Charger la configuration
            Properties config = ConfigLoader.loadConfiguration();
            log.info("Configuration chargée avec succès");
            
            if (args.length > 0) {
                processArguments(args, config);
            } else {
                printUsage();
            }
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution de l'application", e);
            System.exit(1);
        }
        
        log.info("Application terminée avec succès");
    }
    
    private static void processArguments(String[] args, Properties config) throws Exception {
        String command = args[0].toLowerCase();
        
        switch (command) {
            case "train-presence":
                // Vérifier si on utilise YOLO ou le modèle standard
                String presenceModelType = config.getProperty("presence.model.type", "STANDARD");
                
                if ("YOLO".equalsIgnoreCase(presenceModelType)) {
                    log.info("Démarrage de l'entraînement du modèle YOLO de détection de présence");
                    new YOLOPresenceTrainer(config).train();
                } else {
                    log.info("Démarrage de l'entraînement du modèle standard de détection de présence");
                    new PresenceTrainer(config).train();
                }
                break;
                
            case "train-presence-yolo":
                log.info("Démarrage de l'entraînement du modèle YOLO de détection de présence");
                new YOLOPresenceTrainer(config).train();
                break;
                
            case "train-activity":
                // Détecter le type de modèle à utiliser
                String activityModelType = config.getProperty("activity.model.type", "STANDARD");
                
                if ("VGG16".equalsIgnoreCase(activityModelType)) {
                    log.info("Démarrage de l'entraînement du modèle VGG16 de détection d'activité");
                    // Forcer l'utilisation de VGG16
                    config.setProperty("activity.model.type", "VGG16");
                    new ActivityTrainer(config).train();
                } else if ("RESNET".equalsIgnoreCase(activityModelType)) {
                    log.info("Démarrage de l'entraînement du modèle ResNet de détection d'activité");
                    // Forcer l'utilisation de ResNet
                    config.setProperty("activity.model.type", "RESNET");
                    new ActivityTrainer(config).train();
                } else {
                    log.info("Démarrage de l'entraînement du modèle standard de détection d'activité");
                    new ActivityTrainer(config).train();
                }
                break;
                
            case "train-activity-vgg16":
                log.info("Démarrage de l'entraînement du modèle VGG16 de détection d'activité");
                // Forcer l'utilisation de VGG16
                config.setProperty("activity.model.type", "VGG16");
                new ActivityTrainer(config).train();
                break;
                
            case "train-activity-resnet":
                log.info("Démarrage de l'entraînement du modèle ResNet de détection d'activité");
                // Forcer l'utilisation de ResNet
                config.setProperty("activity.model.type", "RESNET");
                new ActivityTrainer(config).train();
                break;
                
            case "train-sound":
                // Détecter le type de modèle à utiliser
                String soundModelType = config.getProperty("sound.model.type", "STANDARD");
                
                if ("SPECTROGRAM".equalsIgnoreCase(soundModelType)) {
                    log.info("Démarrage de l'entraînement du modèle spectrogram de détection de sons");
                    // Forcer l'utilisation de Spectrogram
                    config.setProperty("sound.model.type", "SPECTROGRAM");
                    new SoundTrainer(config).train();
                } else {
                    log.info("Démarrage de l'entraînement du modèle standard de détection de sons");
                    new SoundTrainer(config).train();
                }
                break;
                
            case "train-sound-spectrogram":
                log.info("Démarrage de l'entraînement du modèle spectrogram de détection de sons");
                // Forcer l'utilisation de Spectrogram
                config.setProperty("sound.model.type", "SPECTROGRAM");
                new SoundTrainer(config).train();
                break;
                
            case "train-all":
                log.info("Démarrage de l'entraînement de tous les modèles");
                
                // Entraînement du modèle de présence (YOLO ou standard)
                String presenceType = config.getProperty("presence.model.type", "STANDARD");
                if ("YOLO".equalsIgnoreCase(presenceType)) {
                    new YOLOPresenceTrainer(config).train();
                } else {
                    new PresenceTrainer(config).train();
                }
                
                // Entraînement du modèle d'activité
                String activityType = config.getProperty("activity.model.type", "STANDARD");
                // La classe ActivityTrainer devrait maintenant gérer tous les types de modèles
                new ActivityTrainer(config).train();
                
                // Entraînement du modèle de son
                String soundType = config.getProperty("sound.model.type", "STANDARD");
                // La classe SoundTrainer devrait maintenant gérer tous les types de modèles
                new SoundTrainer(config).train();
                break;
                
            case "export-presence":
                log.info("Exportation du modèle de détection de présence");
                new PresenceExporter(config).export();
                break;
                
            case "export-activity":
                log.info("Exportation du modèle de détection d'activité");
                new ActivityExporter(config).export();
                break;
                
            case "export-sound":
                log.info("Exportation du modèle de détection de sons");
                new SoundExporter(config).export();
                break;
                
            case "export-all":
                log.info("Exportation de tous les modèles");
                new PresenceExporter(config).export();
                new ActivityExporter(config).export();
                new SoundExporter(config).export();
                break;
                
            case "test-presence":
                log.info("Test du modèle de détection de présence");
                validateModel(new ModelValidator(config), "presence");
                break;

            case "test-activity":
                log.info("Test du modèle de détection d'activité");
                validateModel(new ModelValidator(config), "activity");
                break;

            case "test-sound":
                log.info("Test du modèle de détection de sons");
                validateModel(new ModelValidator(config), "sound");
                break;

            case "test-all":
                log.info("Test de tous les modèles");
                validateAllModels(new ModelValidator(config));
                break;
                
            default:
                printUsage();
                break;
        }
    }
    
    /**
     * Valide un modèle spécifique
     * 
     * @param validator Le validateur de modèle
     * @param modelType Le type de modèle à valider
     * @throws IOException Si une erreur survient lors de la validation
     */
    private static void validateModel(ModelValidator validator, String modelType) throws IOException {
        boolean result = false;
        
        switch (modelType) {
            case "presence":
                result = validator.validatePresenceModel();
                break;
            case "activity":
                result = validator.validateActivityModel();
                break;
            case "sound":
                result = validator.validateSoundModel();
                break;
        }
        
        log.info("Validation du modèle {}: {}", modelType, result ? "succès" : "échec");
    }
    
    /**
     * Valide tous les modèles
     * 
     * @param validator Le validateur de modèle
     * @throws IOException Si une erreur survient lors de la validation
     */
    private static void validateAllModels(ModelValidator validator) throws IOException {
        boolean result = validator.validateAllModels();
        log.info("Validation de tous les modèles: {}", result ? "succès" : "échec");
    }
    
    private static void printUsage() {
        System.out.println("Usage: java -jar dl4j-detection-models.jar <commande>");
        System.out.println("Commandes disponibles :");
        System.out.println("  train-presence        : Entraîne le modèle de détection de présence (YOLO ou standard selon la config)");
        System.out.println("  train-presence-yolo   : Entraîne spécifiquement le modèle YOLO de détection de présence");
        System.out.println("  train-activity        : Entraîne le modèle de détection d'activité selon la configuration");
        System.out.println("  train-activity-vgg16  : Entraîne spécifiquement le modèle VGG16 de détection d'activité");
        System.out.println("  train-activity-resnet : Entraîne spécifiquement le modèle ResNet de détection d'activité");
        System.out.println("  train-sound           : Entraîne le modèle de détection de sons selon la configuration");
        System.out.println("  train-sound-spectrogram: Entraîne spécifiquement le modèle basé sur spectrogrammes");
        System.out.println("  train-all             : Entraîne tous les modèles");
        System.out.println("  export-presence       : Exporte le modèle de détection de présence");
        System.out.println("  export-activity       : Exporte le modèle de détection d'activité");
        System.out.println("  export-sound          : Exporte le modèle de détection de sons");
        System.out.println("  export-all            : Exporte tous les modèles");
        System.out.println("  test-presence         : Teste le modèle de détection de présence");
        System.out.println("  test-activity         : Teste le modèle de détection d'activité");
        System.out.println("  test-sound            : Teste le modèle de détection de sons");
        System.out.println("  test-all              : Teste tous les modèles");
    }
}
