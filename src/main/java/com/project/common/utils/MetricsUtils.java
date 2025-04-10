package com.project.common.utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;

/**
 * Classe utilitaire pour simplifier l'utilisation du système de métriques d'évaluation.
 * Fournit des méthodes statiques faciles d'utilisation pour les tâches courantes 
 * liées aux métriques.
 */
public class MetricsUtils {
    private static final Logger log = LoggerFactory.getLogger(MetricsUtils.class);
    
    private MetricsUtils() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Évalue un modèle sur un ensemble de données de test et génère un rapport détaillé
     * 
     * @param model Modèle à évaluer
     * @param testData Données de test
     * @param modelName Nom du modèle pour l'identification des rapports
     * @param config Propriétés de configuration
     * @return Les métriques d'évaluation calculées
     */
    public static EvaluationMetrics evaluateModel(MultiLayerNetwork model, DataSet testData, 
                                               String modelName, Properties config) {
        try {
            ModelEvaluator evaluator = new ModelEvaluator(model, config);
            return evaluator.evaluateAndGenerateReport(testData, modelName);
        } catch (IOException e) {
            log.error("Erreur lors de l'évaluation du modèle : {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Évalue un modèle sur un itérateur de données de test et génère un rapport détaillé
     * 
     * @param model Modèle à évaluer
     * @param testIterator Itérateur sur les données de test
     * @param modelName Nom du modèle pour l'identification des rapports
     * @param config Propriétés de configuration
     * @return Les métriques d'évaluation calculées
     */
    public static EvaluationMetrics evaluateModel(MultiLayerNetwork model, DataSetIterator testIterator, 
                                               String modelName, Properties config) {
        try {
            ModelEvaluator evaluator = new ModelEvaluator(model, config);
            return evaluator.evaluateAndGenerateReport(testIterator, modelName);
        } catch (IOException e) {
            log.error("Erreur lors de l'évaluation du modèle : {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Évalue un modèle sur un ensemble de données de test avec des noms de classes
     * 
     * @param model Modèle à évaluer
     * @param testData Données de test
     * @param modelName Nom du modèle
     * @param labels Liste des noms de classes
     * @param config Propriétés de configuration
     * @return Les métriques d'évaluation calculées
     */
    public static EvaluationMetrics evaluateModelWithLabels(MultiLayerNetwork model, DataSet testData, 
                                                         String modelName, List<String> labels, Properties config) {
        try {
            ModelEvaluator evaluator = new ModelEvaluator(model, config).withLabels(labels);
            return evaluator.evaluateAndGenerateReport(testData, modelName);
        } catch (IOException e) {
            log.error("Erreur lors de l'évaluation du modèle : {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Génère des graphiques pour visualiser les métriques d'un entraînement
     * 
     * @param metrics Liste des métriques collectées pendant l'entraînement
     * @param outputDir Répertoire de sortie pour les graphiques
     * @param modelName Nom du modèle
     */
    public static void generateMetricsCharts(List<EvaluationMetrics> metrics, String outputDir, String modelName) {
        try {
            // Créer le répertoire de sortie s'il n'existe pas
            File dir = new File(outputDir);
            if (!dir.exists() && !dir.mkdirs()) {
                log.warn("Impossible de créer le répertoire de sortie: {}", outputDir);
                return;
            }
            
            // Générer tous les graphiques
            MetricsVisualizer.generateAllCharts(metrics, outputDir, modelName);
            log.info("Graphiques générés avec succès dans {}", outputDir);
        } catch (IOException e) {
            log.error("Erreur lors de la génération des graphiques : {}", e.getMessage());
        }
    }
    
    /**
     * Exporte les métriques au format CSV
     * 
     * @param metrics Liste des métriques collectées pendant l'entraînement
     * @param outputPath Chemin du fichier CSV à créer
     */
    public static void exportMetricsToCSV(List<EvaluationMetrics> metrics, String outputPath) {
        try {
            // Créer le répertoire parent s'il n'existe pas
            File parent = new File(outputPath).getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                log.warn("Impossible de créer le répertoire parent: {}", parent);
                return;
            }
            
            StringBuilder csv = new StringBuilder();
            
            // Écrire l'en-tête
            csv.append("Epoch,Accuracy,Precision,Recall,F1Score,TrainingTime\n");
            
            // Écrire les données
            for (EvaluationMetrics metric : metrics) {
                csv.append(String.format("%d,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                    metric.getEpoch(),
                    metric.getAccuracy(),
                    metric.getPrecision(),
                    metric.getRecall(),
                    metric.getF1Score(),
                    metric.getTrainingTime()
                ));
            }
            
            // Écrire dans le fichier
            Files.writeString(Paths.get(outputPath), csv.toString());
            log.info("Métriques exportées avec succès vers {}", outputPath);
        } catch (IOException e) {
            log.error("Erreur lors de l'exportation des métriques : {}", e.getMessage());
        }
    }
    
    /**
     * Vérifie si les métriques d'évaluation respectent les seuils minimaux définis dans la configuration
     * 
     * @param metrics Métriques d'évaluation à vérifier
     * @param config Propriétés de configuration contenant les seuils
     * @return true si les métriques respectent les seuils, false sinon
     */
    public static boolean validateMetrics(EvaluationMetrics metrics, Properties config) {
        double minAccuracy = Double.parseDouble(config.getProperty("test.min.accuracy", "0.8"));
        double minPrecision = Double.parseDouble(config.getProperty("test.min.precision", "0.75"));
        double minRecall = Double.parseDouble(config.getProperty("test.min.recall", "0.75"));
        double minF1 = Double.parseDouble(config.getProperty("test.min.f1", "0.75"));
        
        boolean valid = true;
        
        if (metrics.getAccuracy() < minAccuracy) {
            log.warn("L'accuracy ({}) est inférieure au seuil minimum ({})", 
                    metrics.getAccuracy(), minAccuracy);
            valid = false;
        }
        
        if (metrics.getPrecision() < minPrecision) {
            log.warn("La precision ({}) est inférieure au seuil minimum ({})", 
                    metrics.getPrecision(), minPrecision);
            valid = false;
        }
        
        if (metrics.getRecall() < minRecall) {
            log.warn("Le recall ({}) est inférieur au seuil minimum ({})", 
                    metrics.getRecall(), minRecall);
            valid = false;
        }
        
        if (metrics.getF1Score() < minF1) {
            log.warn("Le F1-score ({}) est inférieur au seuil minimum ({})", 
                    metrics.getF1Score(), minF1);
            valid = false;
        }
        
        return valid;
    }
    
    /**
     * Génère un rapport de comparaison entre différents modèles
     * 
     * @param metrics Tableau des métriques pour différents modèles
     * @param modelNames Noms des modèles correspondants
     * @param outputPath Chemin du fichier de sortie
     */
    public static void generateModelComparisonReport(EvaluationMetrics[] metrics, 
                                                  String[] modelNames, String outputPath) {
        try {
            if (metrics.length != modelNames.length) {
                throw new IllegalArgumentException("Le nombre de métriques doit correspondre au nombre de noms de modèles");
            }
            
            // Créer le répertoire parent s'il n'existe pas
            File parent = new File(outputPath).getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                log.warn("Impossible de créer le répertoire parent: {}", parent);
                return;
            }
            
            StringBuilder report = new StringBuilder();
            
            // En-tête du rapport
            report.append("===========================================\n");
            report.append("RAPPORT DE COMPARAISON DES MODÈLES\n");
            report.append("===========================================\n\n");
            
            // Tableau de comparaison
            report.append("| Modèle | Accuracy | Precision | Recall | F1-Score | Temps (ms) |\n");
            report.append("|--------|----------|-----------|--------|----------|------------|\n");
            
            for (int i = 0; i < metrics.length; i++) {
                EvaluationMetrics metric = metrics[i];
                String modelName = modelNames[i];
                
                report.append(String.format("| %-6s | %.4f | %.4f | %.4f | %.4f | %.2f |\n",
                    modelName,
                    metric.getAccuracy(),
                    metric.getPrecision(),
                    metric.getRecall(),
                    metric.getF1Score(),
                    metric.getTrainingTime()
                ));
            }
            
            report.append("\n");
            
            // Identifier le meilleur modèle pour chaque métrique
            int bestAccuracyIndex = findBestMetricIndex(metrics, (m) -> m.getAccuracy());
            int bestPrecisionIndex = findBestMetricIndex(metrics, (m) -> m.getPrecision());
            int bestRecallIndex = findBestMetricIndex(metrics, (m) -> m.getRecall());
            int bestF1Index = findBestMetricIndex(metrics, (m) -> m.getF1Score());
            
            report.append("Meilleure Accuracy: ").append(modelNames[bestAccuracyIndex])
                  .append(" (").append(String.format("%.4f", metrics[bestAccuracyIndex].getAccuracy())).append(")\n");
            report.append("Meilleure Precision: ").append(modelNames[bestPrecisionIndex])
                  .append(" (").append(String.format("%.4f", metrics[bestPrecisionIndex].getPrecision())).append(")\n");
            report.append("Meilleur Recall: ").append(modelNames[bestRecallIndex])
                  .append(" (").append(String.format("%.4f", metrics[bestRecallIndex].getRecall())).append(")\n");
            report.append("Meilleur F1-Score: ").append(modelNames[bestF1Index])
                  .append(" (").append(String.format("%.4f", metrics[bestF1Index].getF1Score())).append(")\n");
            
            // Écrire dans le fichier
            Files.writeString(Paths.get(outputPath), report.toString());
            log.info("Rapport de comparaison généré avec succès dans {}", outputPath);
        } catch (IOException e) {
            log.error("Erreur lors de la génération du rapport de comparaison : {}", e.getMessage());
        }
    }
    
    /**
     * Trouve l'index du modèle avec la meilleure valeur pour une métrique donnée
     * 
     * @param metrics Tableau des métriques
     * @param metricExtractor Fonction qui extrait la valeur de métrique à comparer
     * @return Index du modèle avec la meilleure valeur
     */
    private static int findBestMetricIndex(EvaluationMetrics[] metrics, MetricExtractor metricExtractor) {
        int bestIndex = 0;
        double bestValue = metricExtractor.extract(metrics[0]);
        
        for (int i = 1; i < metrics.length; i++) {
            double value = metricExtractor.extract(metrics[i]);
            if (value > bestValue) {
                bestValue = value;
                bestIndex = i;
            }
        }
        
        return bestIndex;
    }
    
    /**
     * Interface fonctionnelle pour extraire une valeur de métrique d'un objet EvaluationMetrics
     */
    @FunctionalInterface
    private interface MetricExtractor {
        double extract(EvaluationMetrics metrics);
    }
}
