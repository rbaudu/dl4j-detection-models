package com.project.common.utils;

import java.util.HashMap;
import java.util.Map;

public class EvaluationMetrics {
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private long timeInMs;
    private int epoch;
    private Map<Integer, ClassMetrics> classMetrics;
    
    public EvaluationMetrics() {
        this.classMetrics = new HashMap<>();
    }
    
    public EvaluationMetrics(double accuracy, double precision, double recall, double f1Score) {
        this();
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
    }
    
    public EvaluationMetrics(double accuracy, double precision, double recall, double f1Score, long timeInMs) {
        this(accuracy, precision, recall, f1Score);
        this.timeInMs = timeInMs;
    }
    
    // Constructeur complet avec epoch
    public EvaluationMetrics(int epoch, double accuracy, double precision, double recall, double f1Score, long timeInMs) {
        this(accuracy, precision, recall, f1Score, timeInMs);
        this.epoch = epoch;
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
    
    public long getTimeInMs() {
        return timeInMs;
    }
    
    public void setTimeInMs(long timeInMs) {
        this.timeInMs = timeInMs;
    }
    
    // Alias pour getTimeInMs pour la compatibilité
    public long getTrainingTime() {
        return timeInMs;
    }
    
    public void setTrainingTime(long timeInMs) {
        this.timeInMs = timeInMs;
    }
    
    public int getEpoch() {
        return epoch;
    }
    
    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }
    
    public Map<Integer, ClassMetrics> getClassMetrics() {
        return classMetrics;
    }
    
    // Alias pour getClassMetrics pour la compatibilité
    public Map<Integer, ClassMetrics> getPerClassMetrics() {
        return classMetrics;
    }
    
    public void setClassMetrics(Map<Integer, ClassMetrics> classMetrics) {
        this.classMetrics = classMetrics;
    }
    
    public void addClassMetrics(int classIndex, ClassMetrics metrics) {
        this.classMetrics.put(classIndex, metrics);
    }
    
    // Surcharge pour addClassMetrics avec les valeurs directes
    public void addClassMetrics(int classIndex, double precision, double recall, double f1Score) {
        ClassMetrics metrics = new ClassMetrics(classIndex, "Class-" + classIndex, precision, recall, f1Score);
        this.classMetrics.put(classIndex, metrics);
    }
    
    /**
     * Formate les métriques en pourcentages pour l'affichage
     */
    private String formatPercent(double value) {
        return String.format("%.2f%%", value * 100);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (epoch > 0) {
            sb.append("Epoch ").append(epoch).append(" - ");
        }
        
        sb.append("Accuracy: ").append(formatPercent(accuracy))
          .append(", Precision: ").append(formatPercent(precision))
          .append(", Recall: ").append(formatPercent(recall))
          .append(", F1: ").append(formatPercent(f1Score));
        
        if (timeInMs > 0) {
            sb.append(", Time: ").append(timeInMs).append(" ms");
        }
        
        return sb.toString();
    }
}