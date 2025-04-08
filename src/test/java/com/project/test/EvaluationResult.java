package com.project.test;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Classe contenant les résultats d'une évaluation de modèle
 */
public class EvaluationResult {
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private Map<String, Double> classAccuracies;
    private Map<String, Object> additionalMetrics;
    
    public EvaluationResult() {
        this.classAccuracies = new LinkedHashMap<>();
        this.additionalMetrics = new LinkedHashMap<>();
    }
    
    // Getters et setters
    public double getAccuracy() {
        return accuracy;
    }
    
    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }
    
    public double getPrecision() {
        return precision;
    }
    
    public void setPrecision(double precision) {
        this.precision = precision;
    }
    
    public double getRecall() {
        return recall;
    }
    
    public void setRecall(double recall) {
        this.recall = recall;
    }
    
    public double getF1Score() {
        return f1Score;
    }
    
    public void setF1Score(double f1Score) {
        this.f1Score = f1Score;
    }
    
    public Map<String, Double> getClassAccuracies() {
        return classAccuracies;
    }
    
    public void addClassAccuracy(String className, double accuracy) {
        this.classAccuracies.put(className, accuracy);
    }
    
    public Map<String, Object> getAdditionalMetrics() {
        return additionalMetrics;
    }
    
    public void addMetric(String name, Object value) {
        this.additionalMetrics.put(name, value);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Évaluation du modèle:\n");
        sb.append("-------------------\n");
        sb.append(String.format("Précision globale: %.2f%%\n", accuracy * 100));
        sb.append(String.format("Précision: %.2f%%\n", precision * 100));
        sb.append(String.format("Rappel: %.2f%%\n", recall * 100));
        sb.append(String.format("Score F1: %.2f%%\n", f1Score * 100));
        
        if (!classAccuracies.isEmpty()) {
            sb.append("\nPrécision par classe:\n");
            for (Map.Entry<String, Double> entry : classAccuracies.entrySet()) {
                sb.append(String.format("  %s: %.2f%%\n", entry.getKey(), entry.getValue() * 100));
            }
        }
        
        if (!additionalMetrics.isEmpty()) {
            sb.append("\nMétriques additionnelles:\n");
            for (Map.Entry<String, Object> entry : additionalMetrics.entrySet()) {
                sb.append(String.format("  %s: %s\n", entry.getKey(), entry.getValue().toString()));
            }
        }
        
        return sb.toString();
    }
}