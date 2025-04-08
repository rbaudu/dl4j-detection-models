package com.project.test;

import com.project.common.config.ConfigLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Classe principale pour exécuter les tests des modèles
 */
public class ModelTestRunner {
    
    private static final Logger log = LoggerFactory.getLogger(ModelTestRunner.class);
    
    private final Properties config;
    private final List<ModelTester> testers;
    
    public ModelTestRunner(Properties config) {
        this.config = config;
        this.testers = new ArrayList<>();
    }
    
    /**
     * Ajoute un testeur de modèle à la liste
     * 
     * @param tester Testeur de modèle à ajouter
     */
    public void addTester(ModelTester tester) {
        testers.add(tester);
    }
    
    /**
     * Exécute les tests pour tous les modèles enregistrés
     * 
     * @return true si tous les tests ont réussi
     */
    public boolean runTests() {
        log.info("Démarrage des tests de modèles...");
        
        boolean allPassed = true;
        int testsRun = 0;
        int testsFailed = 0;
        
        for (ModelTester tester : testers) {
            testsRun++;
            log.info("Test du modèle #{} avec {}", testsRun, tester.getClass().getSimpleName());
            
            // Charger et valider le modèle
            boolean loadSuccess = tester.loadModel();
            if (!loadSuccess) {
                log.error("Échec du chargement du modèle");
                allPassed = false;
                testsFailed++;
                continue;
            }
            
            // Vérifier la validité du modèle
            boolean validationSuccess = tester.validateModel();
            if (!validationSuccess) {
                log.error("Le modèle n'est pas valide");
                allPassed = false;
                testsFailed++;
                continue;
            }
            
            // Tester le modèle sur les données de test
            EvaluationResult result = tester.testModel();
            if (result == null) {
                log.error("Échec de l'évaluation du modèle");
                allPassed = false;
                testsFailed++;
                continue;
            }
            
            // Afficher les résultats
            log.info("Résultats du test:");
            log.info(result.toString());
            
            // Vérifier les critères de réussite
            boolean testPassed = evaluateTestSuccess(result);
            if (!testPassed) {
                log.warn("Le modèle n'a pas satisfait les critères de qualité");
                allPassed = false;
                testsFailed++;
            }
        }
        
        // Résumé des tests
        log.info("Tests terminés. Résumé: {} tests exécutés, {} réussis, {} échoués",
                testsRun, testsRun - testsFailed, testsFailed);
        
        return allPassed;
    }
    
    /**
     * Évalue si les résultats du test satisfont les critères de qualité
     * 
     * @param result Résultats de l'évaluation
     * @return true si le test a réussi
     */
    private boolean evaluateTestSuccess(EvaluationResult result) {
        // Critères de qualité configurables
        double minAccuracy = Double.parseDouble(config.getProperty("test.min.accuracy", "0.7"));
        double minPrecision = Double.parseDouble(config.getProperty("test.min.precision", "0.7"));
        double minRecall = Double.parseDouble(config.getProperty("test.min.recall", "0.7"));
        double minF1Score = Double.parseDouble(config.getProperty("test.min.f1", "0.7"));
        
        // Vérifier les seuils minimaux
        boolean passed = true;
        
        if (result.getAccuracy() < minAccuracy) {
            log.warn("La précision ({}) est inférieure au seuil minimum ({})",
                    result.getAccuracy(), minAccuracy);
            passed = false;
        }
        
        if (result.getPrecision() < minPrecision) {
            log.warn("La précision ({}) est inférieure au seuil minimum ({})",
                    result.getPrecision(), minPrecision);
            passed = false;
        }
        
        if (result.getRecall() < minRecall) {
            log.warn("Le rappel ({}) est inférieur au seuil minimum ({})",
                    result.getRecall(), minRecall);
            passed = false;
        }
        
        if (result.getF1Score() < minF1Score) {
            log.warn("Le score F1 ({}) est inférieur au seuil minimum ({})",
                    result.getF1Score(), minF1Score);
            passed = false;
        }
        
        return passed;
    }
    
    /**
     * Point d'entrée principal pour l'exécution des tests
     */
    public static void main(String[] args) {
        try {
            // Charger la configuration
            Properties config = ConfigLoader.loadConfiguration();
            log.info("Configuration chargée avec succès");
            
            // Créer le runner de test
            ModelTestRunner testRunner = new ModelTestRunner(config);
            
            // Sélection des tests à exécuter
            if (args.length > 0) {
                String modelToTest = args[0].toLowerCase();
                
                switch (modelToTest) {
                    case "presence":
                        testRunner.addTester(new PresenceModelTester(config));
                        break;
                    case "activity":
                        testRunner.addTester(new ActivityModelTester(config));
                        break;
                    case "sound":
                        testRunner.addTester(new SoundModelTester(config));
                        break;
                    case "all":
                        testRunner.addTester(new PresenceModelTester(config));
                        testRunner.addTester(new ActivityModelTester(config));
                        testRunner.addTester(new SoundModelTester(config));
                        break;
                    default:
                        log.error("Modèle non reconnu: {}", modelToTest);
                        printUsage();
                        System.exit(1);
                }
            } else {
                // Par défaut, tester tous les modèles
                testRunner.addTester(new PresenceModelTester(config));
                testRunner.addTester(new ActivityModelTester(config));
                testRunner.addTester(new SoundModelTester(config));
            }
            
            // Exécuter les tests
            boolean success = testRunner.runTests();
            
            // Sortir avec un code approprié
            if (!success) {
                log.error("Certains tests ont échoué");
                System.exit(1);
            } else {
                log.info("Tous les tests ont réussi");
            }
            
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution des tests", e);
            System.exit(1);
        }
    }
    
    private static void printUsage() {
        System.out.println("Usage: java -cp <classpath> com.project.test.ModelTestRunner [model]");
        System.out.println("  où model est l'un des suivants:");
        System.out.println("    presence  - tester uniquement le modèle de détection de présence");
        System.out.println("    activity  - tester uniquement le modèle de détection d'activité");
        System.out.println("    sound     - tester uniquement le modèle de détection de son");
        System.out.println("    all       - tester tous les modèles (par défaut)");
    }
}