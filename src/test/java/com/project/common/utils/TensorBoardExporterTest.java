package com.project.common.utils;

import org.deeplearning4j.ui.model.stats.StatsListener;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe TensorBoardExporter
 */
public class TensorBoardExporterTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private String tensorBoardDir;
    private List<EvaluationMetrics> sampleMetrics;

    @Before
    public void setUp() throws Exception {
        // Créer un répertoire temporaire pour les logs TensorBoard
        File tbDir = tempFolder.newFolder("tensorboard");
        tensorBoardDir = tbDir.getAbsolutePath();

        // Initialiser quelques métriques d'exemple
        sampleMetrics = new ArrayList<>();
        for (int epoch = 1; epoch <= 3; epoch++) {
            // Simuler une amélioration progressive des métriques
            double baseValue = 0.6;
            double improvement = epoch * 0.1;

            EvaluationMetrics metrics = new EvaluationMetrics(
                epoch,
                Math.min(baseValue + improvement, 1.0),
                Math.min(baseValue - 0.05 + improvement, 1.0),
                Math.min(baseValue - 0.1 + improvement, 1.0),
                Math.min(baseValue - 0.08 + improvement, 1.0),
                100.0 * epoch
            );

            // Ajouter quelques métriques par classe
            for (int i = 0; i < 3; i++) {
                metrics.addClassMetrics(
                    i,
                    Math.min(baseValue - 0.05 * i + improvement, 1.0),
                    Math.min(baseValue - 0.1 * i + improvement, 1.0),
                    Math.min(baseValue - 0.08 * i + improvement, 1.0)
                );
            }

            sampleMetrics.add(metrics);
        }
    }

    @After
    public void tearDown() {
        // Arrêter proprement TensorBoard
        TensorBoardExporter.shutdown();
    }

    @Test
    public void testInitialize() {
        // Tester l'initialisation avec un répertoire valide
        boolean result = TensorBoardExporter.initialize(tensorBoardDir);
        assertTrue("L'initialisation devrait réussir avec un répertoire valide", result);

        // Vérifier qu'un fichier de stats est créé dans le répertoire
        File statsFile = new File(tensorBoardDir, "dl4j-stats.bin");
        assertTrue("Un fichier de stats devrait être créé", statsFile.exists());
    }

    @Test
    public void testInitializeInMemory() {
        // Tester l'initialisation en mémoire
        boolean result = TensorBoardExporter.initializeInMemory();
        assertTrue("L'initialisation en mémoire devrait réussir", result);
    }

    @Test
    public void testGetStatsListener() {
        // Initialiser TensorBoard
        TensorBoardExporter.initialize(tensorBoardDir);

        // Récupérer un StatsListener
        StatsListener listener = TensorBoardExporter.getStatsListener("test_model");
        assertNotNull("Un StatsListener devrait être créé", listener);
    }

    @Test
    public void testExportMetrics() {
        // Initialiser TensorBoard
        TensorBoardExporter.initialize(tensorBoardDir);

        // Exporter les métriques
        boolean result = TensorBoardExporter.exportMetrics(sampleMetrics, "test_metrics");
        assertTrue("L'exportation des métriques devrait réussir", result);
    }

    @Test
    public void testExportEvaluation() {
        // Initialiser TensorBoard
        TensorBoardExporter.initialize(tensorBoardDir);
        
        // Créer une évaluation fictive
        Evaluation eval = new Evaluation(3); // 3 classes
        
        // Simuler quelques prédictions pour l'évaluation
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 10; j++) {
                // Simuler 10 prédictions correctes pour chaque classe
                eval.incrementTrue(i);
            }
            
            for (int j = 0; j < 2; j++) {
                // Simuler 2 prédictions incorrectes pour chaque classe
                eval.incrementFalse(i, (i + 1) % 3);
            }
        }
        
        // Exporter l'évaluation
        boolean result = TensorBoardExporter.exportEvaluation(eval, 1, "test_eval");
        assertTrue("L'exportation de l'évaluation devrait réussir", result);
    }
    
    @Test
    public void testExportFromConfig() {
        // Créer une configuration de test
        Properties config = new Properties();
        config.setProperty("tensorboard.log.dir", tensorBoardDir);
        config.setProperty("tensorboard.enabled", "true");
        
        // Exporter les métriques avec la configuration
        boolean result = TensorBoardExporter.exportFromConfig(sampleMetrics, config, "config_test");
        assertTrue("L'exportation depuis la configuration devrait réussir", result);
        
        // Tester avec TensorBoard désactivé
        Properties disabledConfig = new Properties();
        disabledConfig.setProperty("tensorboard.enabled", "false");
        
        result = TensorBoardExporter.exportFromConfig(sampleMetrics, disabledConfig, "disabled_test");
        assertFalse("L'exportation devrait échouer quand TensorBoard est désactivé", result);
    }
    
    @Test
    public void testInitializeWithNonExistentDirectory() {
        // Tester l'initialisation avec un chemin qui nécessite la création de plusieurs répertoires
        String deepPath = tensorBoardDir + "/deep/path/to/logs";
        
        boolean result = TensorBoardExporter.initialize(deepPath);
        assertTrue("L'initialisation devrait créer les répertoires manquants", result);
        
        // Vérifier que les répertoires ont été créés
        File deepDir = new File(deepPath);
        assertTrue("Les répertoires imbriqués devraient être créés", deepDir.exists());
    }
    
    @Test
    public void testMultipleInitialization() {
        // La première initialisation devrait réussir
        boolean result1 = TensorBoardExporter.initialize(tensorBoardDir);
        assertTrue("La première initialisation devrait réussir", result1);
        
        // Une seconde initialisation devrait également réussir
        boolean result2 = TensorBoardExporter.initialize(tensorBoardDir + "/second");
        assertTrue("La seconde initialisation devrait réussir", result2);
    }
    
    @Test
    public void testShutdown() {
        // Initialiser TensorBoard
        TensorBoardExporter.initialize(tensorBoardDir);
        
        // Arrêter TensorBoard
        TensorBoardExporter.shutdown();
        
        // Une nouvelle initialisation devrait réussir après l'arrêt
        boolean result = TensorBoardExporter.initialize(tensorBoardDir);
        assertTrue("L'initialisation après arrêt devrait réussir", result);
    }
    
    @Test
    public void testExportMetricsWithoutInitialization() {
        // Réinitialiser d'abord (pour s'assurer qu'il n'y a pas d'initialisation antérieure)
        TensorBoardExporter.shutdown();
        
        // Exporter des métriques sans initialisation préalable
        // (devrait auto-initialiser en mémoire)
        boolean result = TensorBoardExporter.exportMetrics(sampleMetrics, "auto_init_test");
        assertTrue("L'exportation avec auto-initialisation devrait réussir", result);
    }
}
