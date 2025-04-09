package com.project.test;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Interface pour les testeurs de modèles.
 * Définit les méthodes communes pour charger et tester des modèles.
 */
public interface ModelTester {
    
    /**
     * Charge le modèle à tester
     * 
     * @return true si le chargement a réussi
     */
    boolean loadModel();
    
    /**
     * Effectue des tests sur le modèle
     * 
     * @return résultat de l'évaluation
     */
    EvaluationResult testModel();
    
    /**
     * Vérifie si le modèle est utilisable par une application externe
     * 
     * @return true si le modèle est validé
     */
    boolean validateModel();
}
