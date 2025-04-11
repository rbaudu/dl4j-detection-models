package com.project.common.utils;

import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class ModelEvaluator {
    private static final Logger logger = LoggerFactory.getLogger(ModelEvaluator.class);
    
    private static final double DEFAULT_ACCURACY_THRESHOLD = 0.7;
    private static final double DEFAULT_PRECISION_THRESHOLD = 0.7;
    private static final double DEFAULT_RECALL_THRESHOLD = 0.7;
    private static final double DEFAULT_F1_THRESHOLD = 0.7;
    
    private String modelName;
    private File metricsDir;
    private SimpleDateFormat dateFormat;
    
    public ModelEvaluator(String modelName, String metricsDir) {
        this.modelName = modelName;
        this.metricsDir = new File(metricsDir);
        if (!this.metricsDir.exists()) {
            this.metricsDir.mkdirs();
        }
        this.dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
    }
    
    /**
     * Évalue un modèle avec un ensemble de données et génère un rapport complet
     */
    public EvaluationMetrics evaluateAndGenerateReport(Model model, DataSetIterator testIterator) {
        if (testIterator == null) {
            logger.warn("Iterator de test est null, impossible d'évaluer le modèle");
            return null;
        }
        
        EvaluationMetrics metrics = evaluate(model, testIterator);
        if (metrics == null) {
            return null;
        }
        
        // Vérifier les seuils de qualité
        boolean qualityOk = checkQualityThresholds(metrics);
        
        // Générer le rapport de confusion
        generateConfusionMatrix(model, testIterator);
        
        return metrics;
    }
    
    /**
     * Évalue un modèle avec un ensemble de données
     */
    public EvaluationMetrics evaluate(Model model, DataSetIterator testIterator) {
        if (testIterator == null) {
            logger.warn("Iterator de test est null, impossible d'évaluer le modèle");
            return null;
        }
        
        testIterator.reset();
        long startTime = System.currentTimeMillis();
        
        Evaluation eval = null;
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) model;
            // Obtenir l'évaluation en utilisant l'API disponible
            eval = new Evaluation();
            net.doEvaluation(testIterator, eval);
        } else {
            logger.warn("Type de modèle non supporté: {}", model.getClass().getName());
            return null;
        }
        
        if (eval == null) {
            logger.error("Erreur lors de l'évaluation du modèle");
            return null;
        }
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        // Calculer les métriques globales
        double accuracy = eval.accuracy();
        double precision = eval.precision();
        double recall = eval.recall();
        double f1 = eval.f1();
        
        EvaluationMetrics metrics = new EvaluationMetrics(accuracy, precision, recall, f1, duration);
        
        // Ajouter les métriques par classe (si disponibles)
        try {
            // Utiliser getClasses() qui est une méthode publique
            int classCount = eval.getClasses().size();
            for (int i = 0; i < classCount; i++) {
                String className = "Classe-" + i;
                double classPrecision = eval.precision(i);
                double classRecall = eval.recall(i);
                double classF1 = eval.f1(i);
                
                ClassMetrics classMetrics = new ClassMetrics(i, className, classPrecision, classRecall, classF1);
                
                // Sécurisation de l'accès aux TP/FP/FN selon l'API disponible
                try {
                    // Utiliser les méthodes disponibles dans l'API DL4J 1.0.0-beta7
                    Map<Integer, Integer> tpMap = eval.truePositives();
                    Map<Integer, Integer> fpMap = eval.falsePositives();
                    Map<Integer, Integer> fnMap = eval.falseNegatives();
                    
                    classMetrics.setTruePositives(tpMap.getOrDefault(i, 0));
                    classMetrics.setFalsePositives(fpMap.getOrDefault(i, 0));
                    classMetrics.setFalseNegatives(fnMap.getOrDefault(i, 0));
                } catch (Exception e) {
                    logger.warn("Impossible d'accéder aux TP/FP/FN pour la classe {}: {}", i, e.getMessage());
                }
                
                metrics.addClassMetrics(i, classMetrics);
            }
        } catch (Exception e) {
            logger.warn("Erreur lors de l'accès aux classes de l'évaluation: {}", e.getMessage());
        }
        
        return metrics;
    }
    
    /**
     * Vérifie si les métriques dépassent les seuils de qualité
     */
    public boolean checkQualityThresholds(EvaluationMetrics metrics) {
        return checkQualityThresholds(metrics, 
                                     DEFAULT_ACCURACY_THRESHOLD, 
                                     DEFAULT_PRECISION_THRESHOLD, 
                                     DEFAULT_RECALL_THRESHOLD, 
                                     DEFAULT_F1_THRESHOLD);
    }
    
    /**
     * Vérifie si les métriques dépassent les seuils de qualité spécifiés
     */
    public boolean checkQualityThresholds(EvaluationMetrics metrics, 
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
     * Génère une matrice de confusion pour le modèle
     */
    public void generateConfusionMatrix(Model model, DataSetIterator testIterator) {
        if (testIterator == null) {
            logger.warn("Iterator de test est null, impossible de générer la matrice de confusion");
            return;
        }
        
        testIterator.reset();
        Evaluation eval = new Evaluation();
        
        try {
            if (model instanceof MultiLayerNetwork) {
                MultiLayerNetwork net = (MultiLayerNetwork) model;
                net.doEvaluation(testIterator, eval);
            } else {
                logger.warn("Type de modèle non supporté pour la matrice de confusion: {}", model.getClass().getName());
                return;
            }
            
            // Exporter les résultats de confusion sous forme de texte
            String timestamp = dateFormat.format(new Date());
            File outputFile = new File(metricsDir, modelName + "_confusion_" + timestamp + ".txt");
            
            try (FileWriter writer = new FileWriter(outputFile)) {
                // Exporter sous forme de texte
                writer.write(eval.confusionToString());
                
                logger.info("Matrice de confusion exportée vers {}", outputFile.getAbsolutePath());
            } catch (IOException e) {
                logger.error("Erreur lors de l'exportation de la matrice de confusion : {}", e.getMessage());
            }
        } catch (Exception e) {
            logger.error("Erreur lors de la génération de la matrice de confusion : {}", e.getMessage());
        }
    }
    
    /**
     * Exporte les seuils optimaux pour chaque classe
     */
    public void exportOptimalThresholds(Map<Integer, Double> thresholds) {
        if (thresholds == null || thresholds.isEmpty()) {
            logger.warn("Aucun seuil optimal à exporter");
            return;
        }
        
        String timestamp = dateFormat.format(new Date());
        File outputFile = new File(metricsDir, modelName + "_thresholds_optimal_thresholds_" + timestamp + ".csv");
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            // En-tête
            writer.write("ClassIndex,OptimalThreshold\n");
            
            // Données
            for (Map.Entry<Integer, Double> entry : thresholds.entrySet()) {
                writer.write(String.format("%d,%.4f\n", entry.getKey(), entry.getValue()));
            }
            
            logger.info("Seuils optimaux exportés vers {}", outputFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Erreur lors de l'exportation des seuils optimaux : {}", e.getMessage());
        }
    }
}