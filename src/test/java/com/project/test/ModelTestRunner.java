package com.project.test;

import com.project.common.config.ConfigLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Point d'entrée pour l'exécution des tests des modèles
 */
public class ModelTestRunner {
    private static final Logger log = LoggerFactory.getLogger(ModelTestRunner.class);
    
    public static void main(String[] args) {
        try {
            // Charger la configuration
            Properties config = ConfigLoader.loadConfiguration();
            
            // Déterminer quel modèle tester
            String modelToTest = "all";
            if (args.length > 0) {
                modelToTest = args[0].toLowerCase();
            }
            
            // Configurer les seuils de test
            double minAccuracy = Double.parseDouble(config.getProperty("test.min.accuracy", "0.7"));
            
            boolean allTestsPassed = true;
            
            // Tester les modèles selon le paramètre
            switch (modelToTest) {
                case "presence":
                    allTestsPassed = testPresenceModel(config, minAccuracy);
                    break;
                    
                case "activity":
                    allTestsPassed = testActivityModel(config, minAccuracy);
                    break;
                    
                case "sound":
                    allTestsPassed = testSoundModel(config, minAccuracy);
                    break;
                    
                case "all":
                default:
                    boolean presenceTest = testPresenceModel(config, minAccuracy);
                    boolean activityTest = testActivityModel(config, minAccuracy);
                    boolean soundTest = testSoundModel(config, minAccuracy);
                    allTestsPassed = presenceTest && activityTest && soundTest;
                    break;
            }
            
            // Rapport final
            if (allTestsPassed) {
                log.info("Tous les tests ont réussi!");
                System.exit(0);
            } else {
                log.error("Des tests ont échoué.");
                System.exit(1);
            }
            
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution des tests", e);
            System.exit(1);
        }
    }
    
    /**
     * Teste le modèle de détection de présence
     */
    private static boolean testPresenceModel(Properties config, double minAccuracy) {
        log.info("Test du modèle de détection de présence");
        
        PresenceModelTester tester = new PresenceModelTester(config);
        
        // Charger le modèle
        if (!tester.loadModel()) {
            log.error("Échec du chargement du modèle de détection de présence");
            return false;
        }
        
        // Valider le modèle
        if (!tester.validateModel()) {
            log.error("Le modèle de détection de présence n'est pas valide");
            return false;
        }
        
        // Tester le modèle
        EvaluationResult result = tester.testModel();
        if (result == null) {
            log.error("Échec des tests du modèle de détection de présence");
            return false;
        }
        
        // Vérifier les critères de qualité
        if (result.getAccuracy() < minAccuracy) {
            log.error("Le modèle de détection de présence n'atteint pas la précision minimale requise: {}% < {}%", 
                    result.getAccuracy() * 100, minAccuracy * 100);
            return false;
        }
        
        log.info("Le modèle de détection de présence a passé tous les tests");
        return true;
    }
    
    /**
     * Teste le modèle de détection d'activité
     */
    private static boolean testActivityModel(Properties config, double minAccuracy) {
        log.info("Test du modèle de détection d'activité");
        
        ActivityModelTester tester = new ActivityModelTester(config);
        
        // Charger le modèle
        if (!tester.loadModel()) {
            log.error("Échec du chargement du modèle de détection d'activité");
            return false;
        }
        
        // Valider le modèle
        if (!tester.validateModel()) {
            log.error("Le modèle de détection d'activité n'est pas valide");
            return false;
        }
        
        // Tester le modèle
        EvaluationResult result = tester.testModel();
        if (result == null) {
            log.error("Échec des tests du modèle de détection d'activité");
            return false;
        }
        
        // Vérifier les critères de qualité
        if (result.getAccuracy() < minAccuracy) {
            log.error("Le modèle de détection d'activité n'atteint pas la précision minimale requise: {}% < {}%", 
                    result.getAccuracy() * 100, minAccuracy * 100);
            return false;
        }
        
        log.info("Le modèle de détection d'activité a passé tous les tests");
        return true;
    }
    
    /**
     * Teste le modèle de détection de sons
     */
    private static boolean testSoundModel(Properties config, double minAccuracy) {
        log.info("Test du modèle de détection de sons");
        
        SoundModelTester tester = new SoundModelTester(config);
        
        // Charger le modèle
        if (!tester.loadModel()) {
            log.error("Échec du chargement du modèle de détection de sons");
            return false;
        }
        
        // Valider le modèle
        if (!tester.validateModel()) {
            log.error("Le modèle de détection de sons n'est pas valide");
            return false;
        }
        
        // Tester le modèle
        EvaluationResult result = tester.testModel();
        if (result == null) {
            log.error("Échec des tests du modèle de détection de sons");
            return false;
        }
        
        // Vérifier les critères de qualité
        if (result.getAccuracy() < minAccuracy) {
            log.error("Le modèle de détection de sons n'atteint pas la précision minimale requise: {}% < {}%", 
                    result.getAccuracy() * 100, minAccuracy * 100);
            return false;
        }
        
        log.info("Le modèle de détection de sons a passé tous les tests");
        return true;
    }
}