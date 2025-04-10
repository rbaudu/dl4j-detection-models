package com.project.common.utils;

import java.util.HashMap;
import java.util.Map;

public class EvaluationMetrics {
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private long timeInMs;
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
    
    public Map<Integer, ClassMetrics> getClassMetrics() {
        return classMetrics;
    }
    
    public void setClassMetrics(Map<Integer, ClassMetrics> classMetrics) {
        this.classMetrics = classMetrics;
    }
    
    public void addClassMetrics(int classIndex, ClassMetrics metrics) {
        this.classMetrics.put(classIndex, metrics);
    }
    
    @Override
    public String toString() {
        return String.format("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, Temps: %.2f ms", 
                             accuracy, precision, recall, f1Score, (double)timeInMs);
    }
}