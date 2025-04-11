package com.project.common.utils;

import org.deeplearning4j.ui.model.stats.StatsListener;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.deeplearning4j.eval.Evaluation;

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
    private TensorBoardExporter exporter;

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
                100L * epoch
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
        
        // Créer une instance de TensorBoardExporter pour les tests
        exporter = new TensorBoardExporter(tensorBoardDir);
    }

    @After
    public void tearDown() {
        // Arrêter proprement TensorBoard
        exporter.shutdown();
    }

    @Test
    public void testInitialize() {
        // Tester l'initialisation avec un répertoire valide
        boolean result = exporter.initialize();
        assertTrue("L'initialisation devrait réussir avec un répertoire valide", result);

        // Vérifier qu'un fichier de stats est créé dans le répertoire
        File statsFile = new File(tensorBoardDir, "ui-stats.bin");
        assertTrue("Un fichier de stats devrait être créé", statsFile.exists());
    }

    @Test
    public void testGetStatsListener() {
        // Initialiser TensorBoard
        exporter.initialize();

        // Récupérer un StatsListener
        StatsListener listener = exporter.createListener("test_model");
        assertNotNull("Un StatsListener devrait être créé", listener);
    }

    @Test
    public void testExportMetrics() {
        // Initialiser TensorBoard
        exporter.initialize();

        // Exporter les métriques
        boolean result = TensorBoardExporter.exportMetrics(sampleMetrics, "test_metrics");
        assertTrue("L'exportation des métriques devrait réussir", result);
    }

    @Test
    public void testExportEvaluation() {
        // Initialiser TensorBoard
        exporter.initialize();
        
        // Créer une évaluation fictive
        Evaluation eval = new Evaluation(3); // 3 classes
        
        // Simuler quelques prédictions pour l'évaluation
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 10; j++) {
                // Simuler 10 prédictions correctes pour chaque classe
                eval.incrementTruePositives(i);
            }
            
            for (int j = 0; j < 2; j++) {
                // Simuler 2 prédictions incorrectes pour chaque classe
                eval.incrementFalseNegatives(j);
            }
        }
        
        // Exporter l'évaluation
        boolean result = exporter.exportEvaluation(eval, 1);
        assertTrue("L'exportation de l'évaluation devrait réussir", result);
    }
    
    @Test
    public void testFromConfig() {
        // Créer une configuration de test
        Properties config = new Properties();
        config.setProperty("tensorboard.dir", tensorBoardDir);
        config.setProperty("tensorboard.in.memory", "false");
        
        // Créer un exporter à partir de la configuration
        TensorBoardExporter configExporter = TensorBoardExporter.fromConfig(config);
        assertNotNull("Un exporter devrait être créé à partir de la configuration", configExporter);
        
        // Initialiser l'exporter
        boolean result = configExporter.initialize();
        assertTrue("L'initialisation depuis la configuration devrait réussir", result);
    }
    
    @Test
    public void testInitializeWithNonExistentDirectory() {
        // Tester l'initialisation avec un chemin qui nécessite la création de plusieurs répertoires
        String deepPath = tensorBoardDir + "/deep/path/to/logs";
        
        TensorBoardExporter deepExporter = new TensorBoardExporter(deepPath);
        boolean result = deepExporter.initialize();
        assertTrue("L'initialisation devrait créer les répertoires manquants", result);
        
        // Vérifier que les répertoires ont été créés
        File deepDir = new File(deepPath);
        assertTrue("Les répertoires imbriqués devraient être créés", deepDir.exists());
    }
    
    @Test
    public void testMultipleInitialization() {
        // La première initialisation devrait réussir
        boolean result1 = exporter.initialize();
        assertTrue("La première initialisation devrait réussir", result1);
        
        // Une seconde initialisation devrait également réussir (car déjà initialisé)
        boolean result2 = exporter.initialize();
        assertTrue("La seconde initialisation devrait réussir", result2);
    }
    
    @Test
    public void testShutdown() {
        // Initialiser TensorBoard
        exporter.initialize();
        
        // Arrêter TensorBoard
        exporter.shutdown();
        
        // Une nouvelle initialisation devrait réussir après l'arrêt
        boolean result = exporter.initialize();
        assertTrue("L'initialisation après arrêt devrait réussir", result);
    }
    
    @Test
    public void testExportMetricsWithoutInitialization() {
        // Exporter des métriques sans initialisation préalable
        // (devrait initialiser automatiquement)
        boolean result = TensorBoardExporter.exportMetrics(sampleMetrics, "auto_init_test");
        assertTrue("L'exportation avec auto-initialisation devrait réussir", result);
    }
}