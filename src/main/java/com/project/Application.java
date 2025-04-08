package com.project;

import com.project.common.config.ConfigLoader;
import com.project.export.ActivityExporter;
import com.project.export.PresenceExporter;
import com.project.export.SoundExporter;
import com.project.training.ActivityTrainer;
import com.project.training.PresenceTrainer;
import com.project.training.SoundTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
                log.info("Démarrage de l'entraînement du modèle de détection de présence");
                new PresenceTrainer(config).train();
                break;
                
            case "train-activity":
                log.info("Démarrage de l'entraînement du modèle de détection d'activité");
                new ActivityTrainer(config).train();
                break;
                
            case "train-sound":
                log.info("Démarrage de l'entraînement du modèle de détection de sons");
                new SoundTrainer(config).train();
                break;
                
            case "train-all":
                log.info("Démarrage de l'entraînement de tous les modèles");
                new PresenceTrainer(config).train();
                new ActivityTrainer(config).train();
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
                return new ModelValidator(config).validatePresenceModel();

            case "test-activity":
                log.info("Test du modèle de détection d'activité");
                return new ModelValidator(config).validateActivityModel();

            case "test-sound":
                log.info("Test du modèle de détection de sons");
                return new ModelValidator(config).validateSoundModel();

            case "test-all":
                log.info("Test de tous les modèles");
                return new ModelValidator(config).validateAllModels();                
            default:
                printUsage();
                break;
        }
    }    
    private static void printUsage() {
    	   System.out.println("Usage: java -jar dl4j-detection-models.jar <commande>");
    	    System.out.println("Commandes disponibles :");
    	    System.out.println("  train-presence    : Entraîne le modèle de détection de présence");
    	    System.out.println("  train-activity    : Entraîne le modèle de détection d'activité");
    	    System.out.println("  train-sound       : Entraîne le modèle de détection de sons");
    	    System.out.println("  train-all         : Entraîne tous les modèles");
    	    System.out.println("  export-presence   : Exporte le modèle de détection de présence");
    	    System.out.println("  export-activity   : Exporte le modèle de détection d'activité");
    	    System.out.println("  export-sound      : Exporte le modèle de détection de sons");
    	    System.out.println("  export-all        : Exporte tous les modèles");
    	    System.out.println("  test-presence     : Teste le modèle de détection de présence");
    	    System.out.println("  test-activity     : Teste le modèle de détection d'activité");
    	    System.out.println("  test-sound        : Teste le modèle de détection de sons");
    	    System.out.println("  test-all          : Teste tous les modèles");
    }
}
