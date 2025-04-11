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
            150L    // trainingTime (doit être un long, pas un double)
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
        assertEquals(150L, metrics.getTrainingTime());
    }
    
    @Test
    public void testClassMetrics() {
        // Vérifier que les métriques par classe sont correctement stockées
        Map<Integer, ClassMetrics> perClassMetrics = metrics.getClassMetrics();
        
        // Vérifier que toutes les classes sont présentes
        assertEquals(3, perClassMetrics.size());
        assertTrue(perClassMetrics.containsKey(0));
        assertTrue(perClassMetrics.containsKey(1));
        assertTrue(perClassMetrics.containsKey(2));
        
        // Vérifier les valeurs pour chaque classe
        ClassMetrics class0 = perClassMetrics.get(0);
        assertEquals(0.90, class0.getPrecision(), 0.001);
        assertEquals(0.85, class0.getRecall(), 0.001);
        assertEquals(0.87, class0.getF1Score(), 0.001);
        
        ClassMetrics class1 = perClassMetrics.get(1);
        assertEquals(0.70, class1.getPrecision(), 0.001);
        assertEquals(0.65, class1.getRecall(), 0.001);
        assertEquals(0.67, class1.getF1Score(), 0.001);
    }
    
    @Test
    public void testGetNonExistentClassMetrics() {
        // Vérifier le comportement lors de l'accès à une classe non existante
        assertNull(metrics.getClassMetrics().get(99));
    }
    
    @Test
    public void testToString() {
        // Vérifier que toString génère une chaîne non vide
        String str = metrics.toString();
        assertNotNull(str);
        assertTrue(str.length() > 0);
        
        // Vérifier que la chaîne contient les informations attendues
        // Note: le format exact peut avoir changé
        assertTrue(str.contains("0.85") || str.contains("85%"));  // Accuracy
        assertTrue(str.contains("0.80") || str.contains("80%"));  // Precision
        assertTrue(str.contains("0.75") || str.contains("75%"));  // Recall
        assertTrue(str.contains("0.77") || str.contains("77%"));  // F1
    }
    
    @Test
    public void testAddingAndReplacingClassMetrics() {
        // Ajouter des métriques pour une classe existante (remplace les anciennes)
        metrics.addClassMetrics(1, 0.95, 0.90, 0.92);
        
        // Vérifier que les nouvelles métriques ont été prises en compte
        ClassMetrics class1 = metrics.getClassMetrics().get(1);
        assertEquals(0.95, class1.getPrecision(), 0.001);
        assertEquals(0.90, class1.getRecall(), 0.001);
        assertEquals(0.92, class1.getF1Score(), 0.001);
    }
    
    @Test
    public void testClassMetricsToString() {
        // Vérifier le toString de ClassMetrics
        ClassMetrics classMetrics = metrics.getClassMetrics().get(0);
        String str = classMetrics.toString();
        
        assertNotNull(str);
        assertTrue(str.length() > 0);
        
        // Vérifier que le contenu a bien les valeurs attendues, mais pas forcément le format exact
        assertTrue(str.contains("0.9") || str.contains("90%"));   // Precision
        assertTrue(str.contains("0.85") || str.contains("85%"));  // Recall
        assertTrue(str.contains("0.87") || str.contains("87%"));  // F1
    }
}