package com.project.common.utils;

import org.junit.Before;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe MetricsUtils
 */
public class MetricsUtilsTest {
    
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();
    
    private Properties config;
    private List<EvaluationMetrics> metricsList;
    
    @Before
    public void setUp() {
        // Préparer une configuration de test
        config = new Properties();
        config.setProperty("test.min.accuracy", "0.8");
        config.setProperty("test.min.precision", "0.75");
        config.setProperty("test.min.recall", "0.7");
        config.setProperty("test.min.f1", "0.7");
        
        // Créer une liste de métriques fictives
        metricsList = new ArrayList<>();
        
        // Métriques pour l'époque 1
        EvaluationMetrics metrics1 = new EvaluationMetrics(1, 0.75, 0.73, 0.72, 0.72, 200L);
        metricsList.add(metrics1);
        
        // Métriques pour l'époque 2
        EvaluationMetrics metrics2 = new EvaluationMetrics(2, 0.82, 0.80, 0.78, 0.79, 180L);
        metricsList.add(metrics2);
        
        // Métriques pour l'époque 3
        EvaluationMetrics metrics3 = new EvaluationMetrics(3, 0.88, 0.85, 0.83, 0.84, 190L);
        metricsList.add(metrics3);
    }
    
    @Test
    public void testValidateMetrics() {
        // Métriques qui respectent les seuils
        EvaluationMetrics goodMetrics = new EvaluationMetrics(1, 0.85, 0.80, 0.75, 0.77, 200L);
        assertTrue(MetricsUtils.validateMetrics(goodMetrics, config));
        
        // Métriques qui ne respectent pas l'accuracy minimale
        EvaluationMetrics lowAccuracyMetrics = new EvaluationMetrics(1, 0.75, 0.80, 0.75, 0.77, 200L);
        assertFalse(MetricsUtils.validateMetrics(lowAccuracyMetrics, config));
        
        // Métriques qui ne respectent pas la precision minimale
        EvaluationMetrics lowPrecisionMetrics = new EvaluationMetrics(1, 0.85, 0.70, 0.75, 0.77, 200L);
        assertFalse(MetricsUtils.validateMetrics(lowPrecisionMetrics, config));
        
        // Métriques qui ne respectent pas le recall minimal
        EvaluationMetrics lowRecallMetrics = new EvaluationMetrics(1, 0.85, 0.80, 0.65, 0.77, 200L);
        assertFalse(MetricsUtils.validateMetrics(lowRecallMetrics, config));
        
        // Métriques qui ne respectent pas le F1-score minimal
        EvaluationMetrics lowF1Metrics = new EvaluationMetrics(1, 0.85, 0.80, 0.75, 0.65, 200L);
        assertFalse(MetricsUtils.validateMetrics(lowF1Metrics, config));
    }
    
    @Test
    public void testExportMetricsToCSV() throws IOException {
        // Créer un fichier temporaire pour l'export
        File tempFile = tempFolder.newFile("test_metrics.csv");
        String csvPath = tempFile.getAbsolutePath();
        
        // Exporter les métriques
        MetricsUtils.exportMetricsToCSV(metricsList, csvPath);
        
        // Vérifier que le fichier a été créé
        assertTrue(tempFile.exists());
        
        // Lire le contenu du fichier
        String content = Files.readString(Path.of(csvPath));
        
        // Vérifier le format et le contenu
        assertTrue(content.startsWith("Epoch,Accuracy,Precision,Recall,F1Score,TrainingTime"));
        assertTrue(content.contains("1,0.750000,0.730000,0.720000,0.720000,200"));
        assertTrue(content.contains("2,0.820000,0.800000,0.780000,0.790000,180"));
        assertTrue(content.contains("3,0.880000,0.850000,0.830000,0.840000,190"));
    }
    
    @Test
    public void testGenerateModelComparisonReport() throws IOException {
        // Créer un fichier temporaire pour le rapport
        File tempFile = tempFolder.newFile("test_comparison.txt");
        String reportPath = tempFile.getAbsolutePath();
        
        // Créer des métriques pour différents modèles
        EvaluationMetrics vgg16Metrics = new EvaluationMetrics(100, 0.92, 0.89, 0.91, 0.90, 15000L);
        EvaluationMetrics resnetMetrics = new EvaluationMetrics(100, 0.94, 0.92, 0.90, 0.91, 18000L);
        EvaluationMetrics mobileNetMetrics = new EvaluationMetrics(100, 0.88, 0.87, 0.89, 0.88, 8000L);
        
        // Générer le rapport de comparaison
        MetricsUtils.generateModelComparisonReport(
            new EvaluationMetrics[] { vgg16Metrics, resnetMetrics, mobileNetMetrics },
            new String[] { "VGG16", "ResNet", "MobileNet" },
            reportPath
        );
        
        // Vérifier que le fichier a été créé
        assertTrue(tempFile.exists());
        
        // Lire le contenu du fichier
        String content = Files.readString(Path.of(reportPath));
        
        // Vérifier le contenu du rapport
        assertTrue(content.contains("RAPPORT DE COMPARAISON DES MODÈLES"));
        assertTrue(content.contains("VGG16"));
        assertTrue(content.contains("ResNet"));
        assertTrue(content.contains("MobileNet"));
        
        // Vérifier que le modèle avec la meilleure accuracy est identifié
        assertTrue(content.contains("Meilleure Accuracy: ResNet"));
        
        // Vérifier que le modèle avec la meilleure precision est identifié
        assertTrue(content.contains("Meilleure Precision: ResNet"));
        
        // Vérifier que le modèle avec le meilleur recall est identifié
        assertTrue(content.contains("Meilleur Recall: VGG16"));
        
        // Vérifier que le modèle avec le meilleur F1-score est identifié
        assertTrue(content.contains("Meilleur F1-Score: ResNet"));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGenerateModelComparisonReportWithMismatchedArrays() throws IOException {
        // Créer un fichier temporaire pour le rapport
        File tempFile = tempFolder.newFile("test_mismatch.txt");
        String reportPath = tempFile.getAbsolutePath();
        
        // Métriques et noms de tailles différentes
        EvaluationMetrics[] metrics = new EvaluationMetrics[] {
            new EvaluationMetrics(1, 0.9, 0.9, 0.9, 0.9, 100L),
            new EvaluationMetrics(1, 0.8, 0.8, 0.8, 0.8, 100L)
        };
        
        String[] names = new String[] { "Model1" }; // Un seul nom pour deux métriques
        
        // Cette méthode devrait lever une exception
        MetricsUtils.generateModelComparisonReport(metrics, names, reportPath);
    }
}