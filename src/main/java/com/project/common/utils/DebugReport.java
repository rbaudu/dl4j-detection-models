package com.project.common.utils;

import java.io.IOException;

/**
 * Classe pour déboguer le rapport de comparaison
 */
public class DebugReport {
    public static void main(String[] args) throws IOException {
        // Créer des métriques similaires à celles du test
        EvaluationMetrics vgg16Metrics = new EvaluationMetrics(100, 0.92, 0.89, 0.91, 0.90, 15000L);
        EvaluationMetrics resnetMetrics = new EvaluationMetrics(100, 0.94, 0.92, 0.90, 0.91, 18000L);
        EvaluationMetrics mobileNetMetrics = new EvaluationMetrics(100, 0.88, 0.87, 0.89, 0.88, 8000L);
        
        // Générer le rapport dans un fichier temporaire
        String reportPath = "debug_report.txt";
        
        MetricsUtils.generateModelComparisonReport(
            new EvaluationMetrics[] { vgg16Metrics, resnetMetrics, mobileNetMetrics },
            new String[] { "VGG16", "ResNet", "MobileNet" },
            reportPath
        );
        
        System.out.println("Rapport généré dans: " + reportPath);
        
        // Imprimer quelques valeurs pour voir leur format exact
        System.out.println("Format exact de 0.92: " + vgg16Metrics.getAccuracy());
        System.out.println("Format exact de 0.94: " + resnetMetrics.getAccuracy());
        System.out.println("Format exact de 0.88: " + mobileNetMetrics.getAccuracy());
    }
}