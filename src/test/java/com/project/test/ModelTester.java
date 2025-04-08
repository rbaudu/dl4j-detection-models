package com.project.test;

/**
 * Interface définissant les comportements de test communs pour tous les modèles
 */
public interface ModelTester {
    
    /**
     * Charge un modèle
     * 
     * @return true si le modèle a été chargé avec succès
     */
    boolean loadModel();
    
    /**
     * Teste le modèle sur un ensemble de données de test
     * 
     * @return Résultats de l'évaluation
     */
    EvaluationResult testModel();
    
    /**
     * Vérifie si le modèle est valide
     * 
     * @return true si le modèle est valide pour l'exportation et l'utilisation
     */
    boolean validateModel();
}