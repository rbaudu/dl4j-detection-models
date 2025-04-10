package com.project.common.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class MetricsUtils {
    private static final Logger logger = LoggerFactory.getLogger(MetricsUtils.class);
    
    // Seuils par défaut pour les métriques
    private static final double DEFAULT_ACCURACY_THRESHOLD = 0.7;
    private static final double DEFAULT_PRECISION_THRESHOLD = 0.7;
    private static final double DEFAULT_RECALL_THRESHOLD = 0.7;
    private static final double DEFAULT_F1_THRESHOLD = 0.7;
    
    /**
     * Exporte les métriques vers un fichier CSV
     * 
     * @param metrics Métriques à exporter
     * @param outputPath Chemin du fichier de sortie
     * @return true si l'exportation a réussi, false sinon
     */
    public static boolean exportMetricsToCSV(List<EvaluationMetrics> metrics, String outputPath) {
        if (metrics == null || metrics.isEmpty()) {
            logger.warn("Aucune métrique à exporter");
            return false;
        }
        
        File outputFile = new File(outputPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            // En-tête
            writer.write("Epoch,Accuracy,Precision,Recall,F1,Training_Time_ms\n");
            
            // Données
            for (int i = 0; i < metrics.size(); i++) {
                EvaluationMetrics metric = metrics.get(i);
                writer.write(String.format("%d,%.4f,%.4f,%.4f,%.4f,%d\n", 
                                          i + 1, 
                                          metric.getAccuracy(), 
                                          metric.getPrecision(), 
                                          metric.getRecall(), 
                                          metric.getF1Score(), 
                                          metric.getTimeInMs()));
            }
            
            logger.info("Métriques exportées avec succès vers {}", outputFile.getAbsolutePath());
            return true;
        } catch (IOException e) {
            logger.error("Erreur lors de l'exportation des métriques : {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Vérifie si les métriques sont au-dessus des seuils définis
     * 
     * @param metrics Métriques à vérifier
     * @return true si toutes les métriques sont au-dessus des seuils, false sinon
     */
    public static boolean checkMetricsThresholds(EvaluationMetrics metrics) {
        return checkMetricsThresholds(metrics, 
                                     DEFAULT_ACCURACY_THRESHOLD, 
                                     DEFAULT_PRECISION_THRESHOLD, 
                                     DEFAULT_RECALL_THRESHOLD, 
                                     DEFAULT_F1_THRESHOLD);
    }
    
    /**
     * Vérifie si les métriques sont au-dessus des seuils définis
     * 
     * @param metrics Métriques à vérifier
     * @param accuracyThreshold Seuil d'accuracy
     * @param precisionThreshold Seuil de précision
     * @param recallThreshold Seuil de recall
     * @param f1Threshold Seuil de F1-score
     * @return true si toutes les métriques sont au-dessus des seuils, false sinon
     */
    public static boolean checkMetricsThresholds(EvaluationMetrics metrics,
                                               double accuracyThreshold,
                                               double precisionThreshold,
                                               double recallThreshold,
                                               double f1Threshold) {
        boolean allAboveThresholds = true;
        
        if (metrics.getAccuracy() < accuracyThreshold) {
            logger.warn("L'accuracy ({}) est inférieure au seuil minimum ({})", 
                      metrics.getAccuracy(), accuracyThreshold);
            allAboveThresholds = false;
        }
        
        if (metrics.getPrecision() < precisionThreshold) {
            logger.warn("La precision ({}) est inférieure au seuil minimum ({})", 
                      metrics.getPrecision(), precisionThreshold);
            allAboveThresholds = false;
        }
        
        if (metrics.getRecall() < recallThreshold) {
            logger.warn("Le recall ({}) est inférieur au seuil minimum ({})", 
                      metrics.getRecall(), recallThreshold);
            allAboveThresholds = false;
        }
        
        if (metrics.getF1Score() < f1Threshold) {
            logger.warn("Le F1-score ({}) est inférieur au seuil minimum ({})", 
                      metrics.getF1Score(), f1Threshold);
            allAboveThresholds = false;
        }
        
        return allAboveThresholds;
    }
    
    /**
     * Génère un rapport de comparaison entre deux ensembles de métriques
     * 
     * @param baselineMetrics Métriques de base
     * @param newMetrics Nouvelles métriques
     * @param outputPath Chemin du fichier de sortie
     * @return true si la génération a réussi, false sinon
     */
    public static boolean generateComparisonReport(EvaluationMetrics baselineMetrics, 
                                                 EvaluationMetrics newMetrics, 
                                                 String outputPath) {
        if (baselineMetrics == null || newMetrics == null) {
            logger.warn("Les métriques ne peuvent pas être nulles");
            return false;
        }
        
        File outputFile = new File(outputPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("=== Rapport de comparaison des métriques ===\n\n");
            
            writer.write("Métrique    | Référence | Nouveau | Différence | Amélioration\n");
            writer.write("------------|-----------|---------|------------|-------------\n");
            
            double accuracyDiff = newMetrics.getAccuracy() - baselineMetrics.getAccuracy();
            double precisionDiff = newMetrics.getPrecision() - baselineMetrics.getPrecision();
            double recallDiff = newMetrics.getRecall() - baselineMetrics.getRecall();
            double f1Diff = newMetrics.getF1Score() - baselineMetrics.getF1Score();
            
            writer.write(String.format("Accuracy    | %.4f    | %.4f  | %+.4f      | %s\n", 
                                      baselineMetrics.getAccuracy(), 
                                      newMetrics.getAccuracy(), 
                                      accuracyDiff,
                                      accuracyDiff > 0 ? "Oui" : "Non"));
            
            writer.write(String.format("Precision   | %.4f    | %.4f  | %+.4f      | %s\n", 
                                      baselineMetrics.getPrecision(), 
                                      newMetrics.getPrecision(), 
                                      precisionDiff,
                                      precisionDiff > 0 ? "Oui" : "Non"));
            
            writer.write(String.format("Recall      | %.4f    | %.4f  | %+.4f      | %s\n", 
                                      baselineMetrics.getRecall(), 
                                      newMetrics.getRecall(), 
                                      recallDiff,
                                      recallDiff > 0 ? "Oui" : "Non"));
            
            writer.write(String.format("F1-Score    | %.4f    | %.4f  | %+.4f      | %s\n", 
                                      baselineMetrics.getF1Score(), 
                                      newMetrics.getF1Score(), 
                                      f1Diff,
                                      f1Diff > 0 ? "Oui" : "Non"));
            
            // Comparaison par classe si disponible
            Map<Integer, ClassMetrics> baselineClassMetrics = baselineMetrics.getClassMetrics();
            Map<Integer, ClassMetrics> newClassMetrics = newMetrics.getClassMetrics();
            
            if (!baselineClassMetrics.isEmpty() && !newClassMetrics.isEmpty()) {
                writer.write("\n=== Comparaison des métriques par classe ===\n\n");
                
                for (Integer classIdx : baselineClassMetrics.keySet()) {
                    if (newClassMetrics.containsKey(classIdx)) {
                        ClassMetrics baseline = baselineClassMetrics.get(classIdx);
                        ClassMetrics newer = newClassMetrics.get(classIdx);
                        
                        writer.write(String.format("\nClasse %d (%s):\n", 
                                                 classIdx, 
                                                 baseline.getClassName() != null ? baseline.getClassName() : "Inconnue"));
                        
                        double classPrecisionDiff = newer.getPrecision() - baseline.getPrecision();
                        double classRecallDiff = newer.getRecall() - baseline.getRecall();
                        double classF1Diff = newer.getF1Score() - baseline.getF1Score();
                        
                        writer.write(String.format("Precision: %.4f -> %.4f (%+.4f) %s\n", 
                                                 baseline.getPrecision(), 
                                                 newer.getPrecision(), 
                                                 classPrecisionDiff,
                                                 classPrecisionDiff > 0 ? "✓" : "✗"));
                        
                        writer.write(String.format("Recall: %.4f -> %.4f (%+.4f) %s\n", 
                                                 baseline.getRecall(), 
                                                 newer.getRecall(), 
                                                 classRecallDiff,
                                                 classRecallDiff > 0 ? "✓" : "✗"));
                        
                        writer.write(String.format("F1-Score: %.4f -> %.4f (%+.4f) %s\n", 
                                                 baseline.getF1Score(), 
                                                 newer.getF1Score(), 
                                                 classF1Diff,
                                                 classF1Diff > 0 ? "✓" : "✗"));
                    }
                }
            }
            
            logger.info("Rapport de comparaison généré avec succès dans {}", outputFile.getAbsolutePath());
            return true;
        } catch (IOException e) {
            logger.error("Erreur lors de la génération du rapport de comparaison : {}", e.getMessage());
            return false;
        }
    }
}