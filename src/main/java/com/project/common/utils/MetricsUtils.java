package com.project.common.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;

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
                                          metric.getEpoch() > 0 ? metric.getEpoch() : i + 1, 
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
     * Méthode compatible avec les erreurs dans MetricsExampleUsage
     */
    public static boolean validateMetrics(EvaluationMetrics metrics, Properties config) {
        // Obtenir les seuils depuis la configuration ou utiliser les valeurs par défaut
        double accuracyThreshold = Double.parseDouble(config.getProperty("metrics.threshold.accuracy", "0.7"));
        double precisionThreshold = Double.parseDouble(config.getProperty("metrics.threshold.precision", "0.7"));
        double recallThreshold = Double.parseDouble(config.getProperty("metrics.threshold.recall", "0.7"));
        double f1Threshold = Double.parseDouble(config.getProperty("metrics.threshold.f1", "0.7"));
        
        return checkMetricsThresholds(metrics, accuracyThreshold, precisionThreshold, recallThreshold, f1Threshold);
    }
    
    /**
     * Génère un rapport de comparaison entre plusieurs ensembles de métriques
     */
    public static boolean generateModelComparisonReport(EvaluationMetrics[] metricsArray, 
                                                      String[] modelNames,
                                                      String outputPath) {
        if (metricsArray == null || metricsArray.length == 0 || 
            modelNames == null || modelNames.length != metricsArray.length) {
            logger.warn("Paramètres invalides pour la comparaison de modèles");
            return false;
        }
        
        File outputFile = new File(outputPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("=== Rapport de comparaison des modèles ===\n\n");
            
            writer.write("Modèle        | Accuracy | Precision | Recall  | F1-Score\n");
            writer.write("--------------|----------|-----------|---------|----------\n");
            
            for (int i = 0; i < metricsArray.length; i++) {
                EvaluationMetrics metrics = metricsArray[i];
                writer.write(String.format("%-14s| %.4f   | %.4f    | %.4f   | %.4f\n", 
                                         modelNames[i],
                                         metrics.getAccuracy(), 
                                         metrics.getPrecision(), 
                                         metrics.getRecall(), 
                                         metrics.getF1Score()));
            }
            
            // Identifier le meilleur modèle pour chaque métrique
            int bestAccuracyIdx = findBestModelIndex(metricsArray, m -> m.getAccuracy());
            int bestPrecisionIdx = findBestModelIndex(metricsArray, m -> m.getPrecision());
            int bestRecallIdx = findBestModelIndex(metricsArray, m -> m.getRecall());
            int bestF1Idx = findBestModelIndex(metricsArray, m -> m.getF1Score());
            
            writer.write("\n=== Modèles les plus performants ===\n");
            writer.write("- Meilleure Accuracy:  " + modelNames[bestAccuracyIdx] + " (" + metricsArray[bestAccuracyIdx].getAccuracy() + ")\n");
            writer.write("- Meilleure Precision: " + modelNames[bestPrecisionIdx] + " (" + metricsArray[bestPrecisionIdx].getPrecision() + ")\n");
            writer.write("- Meilleur Recall:     " + modelNames[bestRecallIdx] + " (" + metricsArray[bestRecallIdx].getRecall() + ")\n");
            writer.write("- Meilleur F1-Score:   " + modelNames[bestF1Idx] + " (" + metricsArray[bestF1Idx].getF1Score() + ")\n");
            
            logger.info("Rapport de comparaison généré avec succès dans {}", outputFile.getAbsolutePath());
            return true;
        } catch (IOException e) {
            logger.error("Erreur lors de la génération du rapport de comparaison : {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Trouve l'indice du modèle avec la meilleure valeur pour une métrique donnée
     */
    private static int findBestModelIndex(EvaluationMetrics[] metricsArray, MetricExtractor extractor) {
        int bestIdx = 0;
        double bestValue = extractor.extract(metricsArray[0]);
        
        for (int i = 1; i < metricsArray.length; i++) {
            double value = extractor.extract(metricsArray[i]);
            if (value > bestValue) {
                bestValue = value;
                bestIdx = i;
            }
        }
        
        return bestIdx;
    }
    
    /**
     * Interface fonctionnelle pour extraire une métrique
     */
    @FunctionalInterface
    private interface MetricExtractor {
        double extract(EvaluationMetrics metrics);
    }
    
    /**
     * Génère un rapport de comparaison entre deux ensembles de métriques
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
            
            logger.info("Rapport de comparaison généré avec succès dans {}", outputFile.getAbsolutePath());
            return true;
        } catch (IOException e) {
            logger.error("Erreur lors de la génération du rapport de comparaison : {}", e.getMessage());
            return false;
        }
    }
}