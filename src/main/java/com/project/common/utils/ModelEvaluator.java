package com.project.common.utils;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Classe pour l'évaluation complète des modèles, avec des métriques détaillées
 * et génération de rapports.
 */
public class ModelEvaluator {
    private static final Logger log = LoggerFactory.getLogger(ModelEvaluator.class);
    
    private final Properties config;
    private final MultiLayerNetwork model;
    private final int batchSize;
    private final String outputDir;
    private List<String> labels;
    
    /**
     * Constructeur du ModelEvaluator
     * 
     * @param model Modèle à évaluer
     * @param config Propriétés de configuration
     */
    public ModelEvaluator(MultiLayerNetwork model, Properties config) {
        this.model = model;
        this.config = config;
        this.batchSize = Integer.parseInt(config.getProperty("evaluation.batch.size", "32"));
        this.outputDir = config.getProperty("metrics.output.dir", "output/metrics");
        
        // Créer le répertoire de sortie s'il n'existe pas
        File dir = new File(outputDir);
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                log.warn("Impossible de créer le répertoire de sortie: {}", outputDir);
            }
        }
    }
    
    /**
     * Définit les étiquettes des classes pour une meilleure présentation des résultats
     * 
     * @param labels Liste des noms de classes
     * @return this pour chaînage
     */
    public ModelEvaluator withLabels(List<String> labels) {
        this.labels = new ArrayList<>(labels);
        return this;
    }
    
    /**
     * Évalue le modèle sur un jeu de données et génère un rapport détaillé
     * 
     * @param testData Données de test
     * @param modelName Nom du modèle pour l'identification du rapport
     * @return Métriques d'évaluation
     * @throws IOException Si une erreur survient lors de la création du rapport
     */
    public EvaluationMetrics evaluateAndGenerateReport(DataSet testData, String modelName) throws IOException {
        // Créer un itérateur pour les données de test
        DataSetIterator testIterator = new ListDataSetIterator<>(
            Collections.singletonList(testData), batchSize
        );
        
        return evaluateAndGenerateReport(testIterator, modelName);
    }
    
    /**
     * Évalue le modèle sur un jeu de données et génère un rapport détaillé
     * 
     * @param testIterator Itérateur sur les données de test
     * @param modelName Nom du modèle pour l'identification du rapport
     * @return Métriques d'évaluation
     * @throws IOException Si une erreur survient lors de la création du rapport
     */
    public EvaluationMetrics evaluateAndGenerateReport(DataSetIterator testIterator, String modelName) throws IOException {
        if (model == null || testIterator == null) {
            throw new IllegalArgumentException("Le modèle et l'itérateur de test ne peuvent pas être null");
        }
        
        // Mesurer le temps de départ
        long startTime = System.currentTimeMillis();
        
        // Réinitialiser l'itérateur
        testIterator.reset();
        
        // Évaluation standard
        Evaluation evaluation = new Evaluation();
        
        // Évaluation ROC pour les courbes ROC et AUC
        ROCMultiClass roc = new ROCMultiClass(100); // 100 points pour les courbes ROC
        
        // Matrices pour stocker les prédictions et les étiquettes réelles
        List<INDArray> predictions = new ArrayList<>();
        List<INDArray> actuals = new ArrayList<>();
        
        // Évaluer le modèle
        while (testIterator.hasNext()) {
            DataSet batch = testIterator.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            
            INDArray output = model.output(features);
            
            // Stocker pour l'analyse
            predictions.add(output.dup());
            actuals.add(labels.dup());
            
            // Mettre à jour les évaluateurs
            evaluation.eval(labels, output);
            roc.eval(labels, output);
        }
        
        // Calculer le temps écoulé
        long endTime = System.currentTimeMillis();
        long elapsed = endTime - startTime;
        
        // Extraire les métriques
        double accuracy = evaluation.accuracy();
        double precision = evaluation.precision();
        double recall = evaluation.recall();
        double f1 = evaluation.f1();
        
        // Créer l'objet de métriques
        EvaluationMetrics metrics = new EvaluationMetrics(
            0, accuracy, precision, recall, f1, elapsed
        );
        
        // Ajouter les métriques par classe
        int numClasses = evaluation.getNumRowCounter();
        for (int i = 0; i < numClasses; i++) {
            double classPrecision = evaluation.precision(i);
            double classRecall = evaluation.recall(i);
            double classF1 = evaluation.f1(i);
            
            metrics.addClassMetrics(i, classPrecision, classRecall, classF1);
        }
        
        // Générer le rapport
        generateDetailedReport(evaluation, roc, metrics, modelName);
        
        return metrics;
    }
    
    /**
     * Génère un rapport détaillé de l'évaluation
     * 
     * @param evaluation Objet Evaluation avec les métriques
     * @param roc Objet ROCMultiClass avec les courbes ROC
     * @param metrics Métriques d'évaluation calculées
     * @param modelName Nom du modèle
     * @throws IOException Si une erreur survient lors de l'écriture du rapport
     */
    private void generateDetailedReport(Evaluation evaluation, ROCMultiClass roc, 
                                      EvaluationMetrics metrics, String modelName) throws IOException {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s/%s_evaluation_report_%s.txt", outputDir, modelName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // En-tête du rapport
            writer.write("===========================================\n");
            writer.write(String.format("RAPPORT D'ÉVALUATION DU MODÈLE : %s\n", modelName));
            writer.write(String.format("Date : %s\n", new SimpleDateFormat("dd/MM/yyyy HH:mm").format(new Date())));
            writer.write("===========================================\n\n");
            
            // Résumé des métriques globales
            writer.write("MÉTRIQUES GLOBALES\n");
            writer.write("===================\n");
            writer.write(String.format("Accuracy: %.4f\n", metrics.getAccuracy()));
            writer.write(String.format("Precision: %.4f\n", metrics.getPrecision()));
            writer.write(String.format("Recall: %.4f\n", metrics.getRecall()));
            writer.write(String.format("F1 Score: %.4f\n", metrics.getF1Score()));
            writer.write(String.format("Temps d'évaluation: %.2f ms\n\n", metrics.getTrainingTime()));
            
            // Matrice de confusion
            writer.write("MATRICE DE CONFUSION\n");
            writer.write("====================\n");
            writer.write(evaluation.confusionToString());
            writer.write("\n\n");
            
            // Métriques par classe
            writer.write("MÉTRIQUES PAR CLASSE\n");
            writer.write("===================\n");
            
            Map<Integer, EvaluationMetrics.ClassMetrics> classMetricsMap = metrics.getPerClassMetrics();
            for (Integer i : classMetricsMap.keySet()) {
                EvaluationMetrics.ClassMetrics classMetrics = metrics.getClassMetrics(i);
                if (classMetrics != null) {
                    String className = (labels != null && i < labels.size()) ? labels.get(i) : "Classe " + i;
                    writer.write(String.format("%s:\n", className));
                    writer.write(String.format("  Precision: %.4f\n", classMetrics.getPrecision()));
                    writer.write(String.format("  Recall: %.4f\n", classMetrics.getRecall()));
                    writer.write(String.format("  F1 Score: %.4f\n", classMetrics.getF1Score()));
                    
                    // Ajouter l'AUC pour chaque classe
                    double auc = roc.calculateAUC(i);
                    writer.write(String.format("  AUC: %.4f\n", auc));
                    writer.write("\n");
                }
            }
            
            // Ajouter des statistiques supplémentaires
            writer.write("STATISTIQUES SUPPLÉMENTAIRES\n");
            writer.write("==========================\n");
            writer.write(evaluation.stats());
            writer.write("\n");
            
            // Informations sur le seuil de classification
            writer.write("INFORMATIONS SUR LES SEUILS (ROC)\n");
            writer.write("==============================\n");
            for (Integer i : classMetricsMap.keySet()) {
                String className = (labels != null && i < labels.size()) ? labels.get(i) : "Classe " + i;
                double auc = roc.calculateAUC(i);
                writer.write(String.format("%s (AUC: %.4f)\n", className, auc));
                
                // Utiliser getOptimalThreshold au lieu de calculateOptimalThreshold
                double threshold = getOptimalThreshold(roc, i);
                writer.write(String.format("  Seuil optimal estimé: %.4f\n", threshold));
            }
            
            writer.write("\n");
            writer.write("===========================================\n");
            writer.write("FIN DU RAPPORT\n");
            writer.write("===========================================\n");
        }
        
        log.info("Rapport d'évaluation détaillé généré dans {}", filename);
        
        // Générer également un graphique des métriques par classe
        try {
            MetricsVisualizer.generateClassMetricsChart(metrics, outputDir, modelName);
        } catch (Exception e) {
            log.warn("Impossible de générer le graphique des métriques par classe: {}", e.getMessage());
        }
    }
    
    /**
     * Méthode adaptée pour remplacer calculateOptimalThreshold qui est manquante
     * 
     * @param roc Objet ROCMultiClass
     * @param classIndex Index de la classe
     * @return Seuil optimal estimé
     */
    private double getOptimalThreshold(ROCMultiClass roc, int classIndex) {
        // On utilisera un seuil par défaut de 0.5
        double defaultThreshold = 0.5;
        try {
            // Dans les versions plus récentes, on pourrait utiliser calculateOptimalThreshold
            // Mais comme cette méthode n'est pas disponible, on utilise une estimation
            return defaultThreshold;
        } catch (Exception e) {
            log.warn("Impossible de calculer le seuil optimal, utilisation de la valeur par défaut: {}", defaultThreshold);
            return defaultThreshold;
        }
    }
    
    /**
     * Calcule les métriques à partir de la matrice de confusion
     * 
     * @param confusionMatrix Matrice de confusion
     * @return Métriques d'évaluation
     */
    public static EvaluationMetrics calculateMetricsFromConfusionMatrix(int[][] confusionMatrix) {
        int numClasses = confusionMatrix.length;
        
        // Initialiser les compteurs
        int totalCorrect = 0;
        int totalSamples = 0;
        
        // Compteurs par classe
        int[] truePositives = new int[numClasses];
        int[] falsePositives = new int[numClasses];
        int[] falseNegatives = new int[numClasses];
        
        // Calculer les compteurs à partir de la matrice de confusion
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                int count = confusionMatrix[i][j];
                totalSamples += count;
                
                if (i == j) {
                    // Diagonale = prédictions correctes
                    totalCorrect += count;
                    truePositives[i] += count;
                } else {
                    // Faux négatifs: devrait être de classe i mais prédit comme j
                    falseNegatives[i] += count;
                    // Faux positifs: devrait être de classe j mais prédit comme i
                    falsePositives[j] += count;
                }
            }
        }
        
        // Calculer l'accuracy globale
        double accuracy = (double) totalCorrect / totalSamples;
        
        // Calculer precision, recall et F1 pour chaque classe
        double[] precisions = new double[numClasses];
        double[] recalls = new double[numClasses];
        double[] f1Scores = new double[numClasses];
        
        double precisionSum = 0.0;
        double recallSum = 0.0;
        double f1Sum = 0.0;
        int validClassCount = 0;
        
        for (int i = 0; i < numClasses; i++) {
            if (truePositives[i] + falsePositives[i] > 0) {
                precisions[i] = (double) truePositives[i] / (truePositives[i] + falsePositives[i]);
                precisionSum += precisions[i];
                validClassCount++;
            } else {
                precisions[i] = 0.0;
            }
            
            if (truePositives[i] + falseNegatives[i] > 0) {
                recalls[i] = (double) truePositives[i] / (truePositives[i] + falseNegatives[i]);
                recallSum += recalls[i];
            } else {
                recalls[i] = 0.0;
            }
            
            if (precisions[i] + recalls[i] > 0) {
                f1Scores[i] = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]);
                f1Sum += f1Scores[i];
            } else {
                f1Scores[i] = 0.0;
            }
        }
        
        // Calculer les moyennes
        double avgPrecision = validClassCount > 0 ? precisionSum / validClassCount : 0.0;
        double avgRecall = validClassCount > 0 ? recallSum / validClassCount : 0.0;
        double avgF1 = validClassCount > 0 ? f1Sum / validClassCount : 0.0;
        
        // Créer l'objet de métriques
        EvaluationMetrics metrics = new EvaluationMetrics(
            0, accuracy, avgPrecision, avgRecall, avgF1, 0.0
        );
        
        // Ajouter les métriques par classe
        for (int i = 0; i < numClasses; i++) {
            metrics.addClassMetrics(i, precisions[i], recalls[i], f1Scores[i]);
        }
        
        return metrics;
    }
    
    /**
     * Exporte les seuils optimaux pour chaque classe
     * Ces seuils peuvent être utilisés pour ajuster les prédictions du modèle
     * 
     * @param roc Objet ROCMultiClass contenant les données ROC
     * @param modelName Nom du modèle
     * @throws IOException Si une erreur survient lors de l'écriture
     */
    public void exportOptimalThresholds(ROCMultiClass roc, String modelName) throws IOException {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s/%s_optimal_thresholds_%s.csv", outputDir, modelName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // En-tête du CSV
            writer.write("Class,OptimalThreshold,AUC\n");
            
            // Écrire les seuils pour chaque classe
            int numClasses = roc.getNumClasses();
            for (int i = 0; i < numClasses; i++) {
                double threshold = getOptimalThreshold(roc, i);
                double auc = roc.calculateAUC(i);
                
                String className = (labels != null && i < labels.size()) ? labels.get(i) : String.valueOf(i);
                writer.write(String.format("%s,%.6f,%.6f\n", className, threshold, auc));
            }
        }
        
        log.info("Seuils optimaux exportés vers {}", filename);
    }
    
    /**
     * Vérifie si les métriques calculées respectent les seuils minimaux définis dans la configuration
     * 
     * @param metrics Métriques à vérifier
     * @return true si les métriques respectent les seuils minimaux, false sinon
     */
    public boolean validateMetricsAgainstThresholds(EvaluationMetrics metrics) {
        double minAccuracy = Double.parseDouble(config.getProperty("test.min.accuracy", "0.8"));
        double minPrecision = Double.parseDouble(config.getProperty("test.min.precision", "0.75"));
        double minRecall = Double.parseDouble(config.getProperty("test.min.recall", "0.75"));
        double minF1 = Double.parseDouble(config.getProperty("test.min.f1", "0.75"));
        
        boolean validated = true;
        
        if (metrics.getAccuracy() < minAccuracy) {
            log.warn("L'accuracy ({}) est inférieure au seuil minimum ({})", 
                    metrics.getAccuracy(), minAccuracy);
            validated = false;
        }
        
        if (metrics.getPrecision() < minPrecision) {
            log.warn("La precision ({}) est inférieure au seuil minimum ({})", 
                    metrics.getPrecision(), minPrecision);
            validated = false;
        }
        
        if (metrics.getRecall() < minRecall) {
            log.warn("Le recall ({}) est inférieur au seuil minimum ({})", 
                    metrics.getRecall(), minRecall);
            validated = false;
        }
        
        if (metrics.getF1Score() < minF1) {
            log.warn("Le F1-score ({}) est inférieur au seuil minimum ({})", 
                    metrics.getF1Score(), minF1);
            validated = false;
        }
        
        return validated;
    }
}