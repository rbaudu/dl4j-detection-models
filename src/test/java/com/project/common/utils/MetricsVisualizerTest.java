package com.project.common.utils;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe MetricsVisualizer
 */
public class MetricsVisualizerTest {
    
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();
    
    private List<EvaluationMetrics> metricsList;
    private String outputDir;
    
    @Before
    public void setUp() throws IOException {
        // Créer un répertoire de sortie temporaire
        File dir = tempFolder.newFolder("metrics_visualizer_test");
        outputDir = dir.getAbsolutePath();
        
        // Créer une liste de métriques fictives
        metricsList = new ArrayList<>();
        
        // Métriques pour 5 époques avec une progression
        for (int epoch = 1; epoch <= 5; epoch++) {
            // Simuler une amélioration progressive des métriques
            double baseAccuracy = 0.70;
            double basePrecision = 0.65;
            double baseRecall = 0.60;
            double baseF1 = 0.62;
            
            double improvement = epoch * 0.05; // 5% d'amélioration par époque
            
            EvaluationMetrics metrics = new EvaluationMetrics(
                epoch,
                Math.min(baseAccuracy + improvement, 1.0),
                Math.min(basePrecision + improvement, 1.0),
                Math.min(baseRecall + improvement, 1.0),
                Math.min(baseF1 + improvement, 1.0),
                150L + (epoch * 10L) // Le temps d'entraînement augmente un peu à chaque époque
            );
            
            // Ajouter des métriques par classe
            for (int i = 0; i < 3; i++) {
                metrics.addClassMetrics(
                    i, 
                    Math.min(0.6 + (i * 0.1) + improvement, 1.0),
                    Math.min(0.55 + (i * 0.1) + improvement, 1.0),
                    Math.min(0.58 + (i * 0.1) + improvement, 1.0)
                );
            }
            
            metricsList.add(metrics);
        }
    }
    
    @Test
    public void testGenerateEvolutionChart() throws IOException {
        // Générer le graphique d'évolution
        MetricsVisualizer.generateEvolutionChart(metricsList, outputDir, "test_model");
        
        // Vérifier que le fichier a été créé
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.startsWith("test_model_metrics_evolution_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertTrue(files.length > 0);
        
        // Vérifier la taille du fichier (il devrait être non vide)
        assertTrue(files[0].length() > 0);
    }
    
    @Test
    public void testGenerateClassMetricsChart() throws IOException {
        // Obtenir la dernière métrique pour le test
        EvaluationMetrics lastMetrics = metricsList.get(metricsList.size() - 1);
        
        // Générer le graphique des métriques par classe
        MetricsVisualizer.generateClassMetricsChart(lastMetrics, outputDir, "test_model");
        
        // Vérifier que le fichier a été créé
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.startsWith("test_model_class_metrics_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertTrue(files.length > 0);
        
        // Vérifier la taille du fichier (il devrait être non vide)
        assertTrue(files[0].length() > 0);
    }
    
    @Test
    public void testGenerateTrainingTimeChart() throws IOException {
        // Générer le graphique des temps d'entraînement
        MetricsVisualizer.generateTrainingTimeChart(metricsList, outputDir, "test_model");
        
        // Vérifier que le fichier a été créé
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.startsWith("test_model_training_time_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertTrue(files.length > 0);
        
        // Vérifier la taille du fichier (il devrait être non vide)
        assertTrue(files[0].length() > 0);
    }
    
    @Test
    public void testGenerateAllCharts() throws IOException {
        // Générer tous les graphiques
        MetricsVisualizer.generateAllCharts(metricsList, outputDir, "test_model");
        
        // Vérifier que les fichiers ont été créés (au moins 3 graphiques)
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.startsWith("test_model_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertTrue(files.length >= 3);
        
        // Vérifier que chaque type de graphique a été généré
        boolean hasEvolutionChart = false;
        boolean hasClassChart = false;
        boolean hasTimeChart = false;
        
        for (File file : files) {
            String name = file.getName();
            if (name.contains("evolution")) {
                hasEvolutionChart = true;
            } else if (name.contains("class_metrics")) {
                hasClassChart = true;
            } else if (name.contains("training_time")) {
                hasTimeChart = true;
            }
        }
        
        assertTrue("Le graphique d'évolution devrait être généré", hasEvolutionChart);
        assertTrue("Le graphique des métriques par classe devrait être généré", hasClassChart);
        assertTrue("Le graphique des temps d'entraînement devrait être généré", hasTimeChart);
    }
    
    @Test
    public void testEmptyMetricsList() throws IOException {
        // Tester avec une liste vide
        List<EvaluationMetrics> emptyList = new ArrayList<>();
        
        // Aucune exception ne devrait être levée, mais aucun fichier ne devrait être créé
        MetricsVisualizer.generateAllCharts(emptyList, outputDir, "empty_test");
        
        // Vérifier qu'aucun fichier n'a été créé
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.startsWith("empty_test_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertEquals(0, files.length);
    }
    
    @Test
    public void testDirectoryCreation() throws IOException {
        // Créer un chemin de répertoire qui n'existe pas
        String nonExistentDir = outputDir + "/subdir/nonexistent";
        
        // La méthode devrait créer le répertoire s'il n'existe pas
        MetricsVisualizer.generateEvolutionChart(metricsList, nonExistentDir, "test_model");
        
        // Vérifier que le répertoire a été créé
        File dir = new File(nonExistentDir);
        assertTrue(dir.exists());
        
        // Vérifier que le fichier a été créé
        File[] files = dir.listFiles((d, name) -> 
            name.startsWith("test_model_metrics_evolution_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertTrue(files.length > 0);
    }
    
    @Test
    public void testSingleMetric() throws IOException {
        // Tester avec une seule métrique
        List<EvaluationMetrics> singleMetricList = new ArrayList<>();
        singleMetricList.add(metricsList.get(0));
        
        // Les graphiques d'évolution et de temps ne sont pas très utiles avec une seule époque,
        // mais ils devraient quand même être générés
        MetricsVisualizer.generateAllCharts(singleMetricList, outputDir, "single_test");
        
        // Vérifier que les fichiers ont été créés
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.startsWith("single_test_") && name.endsWith(".png"));
            
        assertNotNull(files);
        assertTrue(files.length >= 3);
    }
}