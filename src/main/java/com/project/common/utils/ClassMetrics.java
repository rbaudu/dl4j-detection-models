package com.project.common.utils;

/**
 * Classe représentant les métriques d'évaluation pour une classe spécifique
 */
public class ClassMetrics {
    private int classIndex;
    private String className;
    private double precision;
    private double recall;
    private double f1Score;
    private int truePositives;
    private int falsePositives;
    private int falseNegatives;
    
    /**
     * Constructeur par défaut
     */
    public ClassMetrics() {
    }
    
    /**
     * Constructeur avec index de classe et nom
     */
    public ClassMetrics(int classIndex, String className) {
        this.classIndex = classIndex;
        this.className = className;
    }
    
    /**
     * Constructeur complet
     */
    public ClassMetrics(int classIndex, String className, double precision, double recall, double f1Score) {
        this.classIndex = classIndex;
        this.className = className;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
    }
    
    /**
     * Constructeur complet avec statistiques détaillées
     */
    public ClassMetrics(int classIndex, String className, double precision, double recall, double f1Score,
                        int truePositives, int falsePositives, int falseNegatives) {
        this.classIndex = classIndex;
        this.className = className;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
        this.truePositives = truePositives;
        this.falsePositives = falsePositives;
        this.falseNegatives = falseNegatives;
    }
    
    // Getters et setters
    
    public int getClassIndex() {
        return classIndex;
    }
    
    public void setClassIndex(int classIndex) {
        this.classIndex = classIndex;
    }
    
    public String getClassName() {
        return className;
    }
    
    public void setClassName(String className) {
        this.className = className;
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
    
    public int getTruePositives() {
        return truePositives;
    }
    
    public void setTruePositives(int truePositives) {
        this.truePositives = truePositives;
    }
    
    public int getFalsePositives() {
        return falsePositives;
    }
    
    public void setFalsePositives(int falsePositives) {
        this.falsePositives = falsePositives;
    }
    
    public int getFalseNegatives() {
        return falseNegatives;
    }
    
    public void setFalseNegatives(int falseNegatives) {
        this.falseNegatives = falseNegatives;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (className != null && !className.isEmpty()) {
            sb.append("Class ").append(classIndex).append(" [").append(className).append("] - ");
        } else {
            sb.append("Class ").append(classIndex).append(" - ");
        }
        
        // Format avec les valeurs décimales exactes
        sb.append("Precision: ").append(precision)
          .append(", Recall: ").append(recall)
          .append(", F1: ").append(f1Score);
        
        if (truePositives > 0 || falsePositives > 0 || falseNegatives > 0) {
            sb.append(", TP: ").append(truePositives)
              .append(", FP: ").append(falsePositives)
              .append(", FN: ").append(falseNegatives);
        }
        
        return sb.toString();
    }
}