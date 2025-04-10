package com.project.common.utils;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Classe pour stocker les métriques d'évaluation (précision, rappel, F1-score) d'un modèle
 * pour toutes les classes et la moyenne globale.
 */
public class EvaluationMetrics implements Serializable {

    private static final long serialVersionUID = 1L;
    
    private int epoch;
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private double trainingTime;
    private Map<Integer, ClassMetrics> perClassMetrics;
    
    /**
     * Constructeur pour l'initialisation des métriques
     * 
     * @param epoch Numéro de l'époque
     * @param accuracy Précision globale (accuracy)
     * @param precision Précision moyenne sur toutes les classes
     * @param recall Rappel moyen sur toutes les classes
     * @param f1Score Score F1 moyen sur toutes les classes
     * @param trainingTime Temps d'entraînement pour cette époque (en ms)
     */
    public EvaluationMetrics(int epoch, double accuracy, double precision, 
                            double recall, double f1Score, double trainingTime) {
        this.epoch = epoch;
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
        this.trainingTime = trainingTime;
        this.perClassMetrics = new HashMap<>();
    }
    
    /**
     * Classe interne pour stocker les métriques d'une classe spécifique
     */
    public static class ClassMetrics implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private double precision;
        private double recall;
        private double f1Score;
        
        public ClassMetrics(double precision, double recall, double f1Score) {
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
        }
        
        public double getPrecision() { return precision; }
        public double getRecall() { return recall; }
        public double getF1Score() { return f1Score; }
        
        @Override
        public String toString() {
            return String.format("Précision: %.4f, Rappel: %.4f, F1-Score: %.4f", 
                                 precision, recall, f1Score);
        }
    }

    /**
     * Ajoute les métriques pour une classe spécifique
     * 
     * @param classIndex Indice de la classe
     * @param precision Précision pour cette classe
     * @param recall Rappel pour cette classe
     * @param f1Score Score F1 pour cette classe
     */
    public void addClassMetrics(int classIndex, double precision, double recall, double f1Score) {
        perClassMetrics.put(classIndex, new ClassMetrics(precision, recall, f1Score));
    }
    
    /**
     * Obtient les métriques pour une classe spécifique
     * 
     * @param classIndex Indice de la classe
     * @return Métriques de la classe ou null si non disponible
     */
    public ClassMetrics getClassMetrics(int classIndex) {
        return perClassMetrics.get(classIndex);
    }
    
    // Getters
    public int getEpoch() { return epoch; }
    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
    public double getTrainingTime() { return trainingTime; }
    public Map<Integer, ClassMetrics> getPerClassMetrics() { return perClassMetrics; }
    
    @Override
    public String toString() {
        return String.format("Époque %d - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, Temps: %.2f ms",
                            epoch, accuracy, precision, recall, f1Score, trainingTime);
    }
    
    /**
     * Génère un rapport détaillé des métriques
     * 
     * @return Rapport formaté des métriques
     */
    public String generateDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("=== Rapport d'évaluation pour l'époque %d ===\n", epoch));
        sb.append(String.format("Accuracy globale: %.4f\n", accuracy));
        sb.append(String.format("Precision moyenne: %.4f\n", precision));
        sb.append(String.format("Recall moyen: %.4f\n", recall));
        sb.append(String.format("F1-Score moyen: %.4f\n", f1Score));
        sb.append(String.format("Temps d'entraînement: %.2f ms\n", trainingTime));
        sb.append("\nMétriques par classe:\n");
        
        for (Map.Entry<Integer, ClassMetrics> entry : perClassMetrics.entrySet()) {
            ClassMetrics metrics = entry.getValue();
            sb.append(String.format("Classe %d: %s\n", entry.getKey(), metrics.toString()));
        }
        
        return sb.toString();
    }
}
