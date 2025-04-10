package com.project.common.utils;

import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe EvaluationMetrics
 */
public class EvaluationMetricsTest {
    
    private EvaluationMetrics metrics;
    
    @Before
    public void setUp() {
        // Créer une instance de test avec des valeurs connues
        metrics = new EvaluationMetrics(
            1,      // epoch
            0.85,   // accuracy
            0.80,   // precision
            0.75,   // recall
            0.77,   // f1Score
            150.0   // trainingTime
        );
        
        // Ajouter des métriques pour quelques classes
        metrics.addClassMetrics(0, 0.90, 0.85, 0.87);
        metrics.addClassMetrics(1, 0.70, 0.65, 0.67);
        metrics.addClassMetrics(2, 0.80, 0.75, 0.77);
    }
    
    @Test
    public void testGetters() {
        // Vérifier que les getters retournent les valeurs correctes
        assertEquals(1, metrics.getEpoch());
        assertEquals(0.85, metrics.getAccuracy(), 0.001);
        assertEquals(0.80, metrics.getPrecision(), 0.001);
        assertEquals(0.75, metrics.getRecall(), 0.001);
        assertEquals(0.77, metrics.getF1Score(), 0.001);
        assertEquals(150.0, metrics.getTrainingTime(), 0.001);
    }
    
    @Test
    public void testClassMetrics() {
        // Vérifier que les métriques par classe sont correctement stockées
        Map<Integer, EvaluationMetrics.ClassMetrics> perClassMetrics = metrics.getPerClassMetrics();
        
        // Vérifier que toutes les classes sont présentes
        assertEquals(3, perClassMetrics.size());
        assertTrue(perClassMetrics.containsKey(0));
        assertTrue(perClassMetrics.containsKey(1));
        assertTrue(perClassMetrics.containsKey(2));
        
        // Vérifier les valeurs pour chaque classe
        EvaluationMetrics.ClassMetrics class0 = metrics.getClassMetrics(0);
        assertEquals(0.90, class0.getPrecision(), 0.001);
        assertEquals(0.85, class0.getRecall(), 0.001);
        assertEquals(0.87, class0.getF1Score(), 0.001);
        
        EvaluationMetrics.ClassMetrics class1 = metrics.getClassMetrics(1);
        assertEquals(0.70, class1.getPrecision(), 0.001);
        assertEquals(0.65, class1.getRecall(), 0.001);
        assertEquals(0.67, class1.getF1Score(), 0.001);
    }
    
    @Test
    public void testGetNonExistentClassMetrics() {
        // Vérifier le comportement lors de l'accès à une classe non existante
        assertNull(metrics.getClassMetrics(99));
    }
    
    @Test
    public void testToString() {
        // Vérifier que toString génère une chaîne non vide
        String str = metrics.toString();
        assertNotNull(str);
        assertTrue(str.length() > 0);
        
        // Vérifier que la chaîne contient les informations attendues
        assertTrue(str.contains("Epoch 1"));
        assertTrue(str.contains("0.85"));  // Accuracy
        assertTrue(str.contains("0.80"));  // Precision
        assertTrue(str.contains("0.75"));  // Recall
        assertTrue(str.contains("0.77"));  // F1
    }
    
    @Test
    public void testGenerateDetailedReport() {
        // Vérifier que le rapport détaillé est généré correctement
        String report = metrics.generateDetailedReport();
        assertNotNull(report);
        assertTrue(report.length() > 0);
        
        // Vérifier que le rapport contient les informations attendues
        assertTrue(report.contains("Rapport d'évaluation pour l'époque 1"));
        assertTrue(report.contains("Accuracy globale:"));
        assertTrue(report.contains("Precision moyenne:"));
        assertTrue(report.contains("Recall moyen:"));
        assertTrue(report.contains("F1-Score moyen:"));
        assertTrue(report.contains("Métriques par classe:"));
        assertTrue(report.contains("Classe 0:"));
        assertTrue(report.contains("Classe 1:"));
        assertTrue(report.contains("Classe 2:"));
    }
    
    @Test
    public void testAddingAndReplacingClassMetrics() {
        // Ajouter des métriques pour une classe existante (remplace les anciennes)
        metrics.addClassMetrics(1, 0.95, 0.90, 0.92);
        
        // Vérifier que les nouvelles métriques ont été prises en compte
        EvaluationMetrics.ClassMetrics class1 = metrics.getClassMetrics(1);
        assertEquals(0.95, class1.getPrecision(), 0.001);
        assertEquals(0.90, class1.getRecall(), 0.001);
        assertEquals(0.92, class1.getF1Score(), 0.001);
    }
    
    @Test
    public void testClassMetricsToString() {
        // Vérifier le toString de ClassMetrics
        EvaluationMetrics.ClassMetrics classMetrics = metrics.getClassMetrics(0);
        String str = classMetrics.toString();
        
        assertNotNull(str);
        assertTrue(str.length() > 0);
        assertTrue(str.contains("Précision: 0.9"));
        assertTrue(str.contains("Rappel: 0.85"));
        assertTrue(str.contains("F1-Score: 0.87"));
    }
}
