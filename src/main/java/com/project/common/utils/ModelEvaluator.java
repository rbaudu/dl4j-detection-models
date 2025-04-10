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
import java.util.List;
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
        
        // Générer la courbe ROC si binaire ou multi-classe
        generateROCCurve(model, testIterator);
        
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
        
        Evaluation eval = new Evaluation();
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) model;
            net.evaluate(testIterator, eval);
        } else {
            logger.warn("Type de modèle non supporté: {}", model.getClass().getName());
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
        int numClasses = eval.getClasses().size();
        for (int i = 0; i < numClasses; i++) {
            String className = eval.getClassLabel(i);
            double classPrecision = eval.precision(i);
            double classRecall = eval.recall(i);
            double classF1 = eval.f1(i);
            
            ClassMetrics classMetrics = new ClassMetrics(i, className, classPrecision, classRecall, classF1);
            classMetrics.setTruePositives((int) eval.truePositives().getDouble(i));
            classMetrics.setFalsePositives((int) eval.falsePositives().getDouble(i));
            classMetrics.setFalseNegatives((int) eval.falseNegatives().getDouble(i));
            
            metrics.addClassMetrics(i, classMetrics);
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
        
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) model;
            net.evaluate(testIterator, eval);
        } else {
            logger.warn("Type de modèle non supporté pour la matrice de confusion: {}", model.getClass().getName());
            return;
        }
        
        // Récupérer la matrice de confusion
        INDArray confusionMatrix = eval.getConfusionMatrix();
        
        // Exporter la matrice de confusion
        String timestamp = dateFormat.format(new Date());
        File outputFile = new File(metricsDir, modelName + "_confusion_matrix_" + timestamp + ".csv");
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            // En-tête
            writer.write("PredictedClass");
            for (int i = 0; i < eval.numClasses(); i++) {
                writer.write("," + eval.getClassLabel(i));
            }
            writer.write("\n");
            
            // Contenu
            for (int i = 0; i < eval.numClasses(); i++) {
                writer.write(eval.getClassLabel(i));
                for (int j = 0; j < eval.numClasses(); j++) {
                    writer.write("," + (int) confusionMatrix.getDouble(i, j));
                }
                writer.write("\n");
            }
            
            logger.info("Matrice de confusion exportée vers {}", outputFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Erreur lors de l'exportation de la matrice de confusion : {}", e.getMessage());
        }
    }
    
    /**
     * Génère des courbes ROC pour le modèle
     */
    public void generateROCCurve(Model model, DataSetIterator testIterator) {
        if (testIterator == null) {
            logger.warn("Iterator de test est null, impossible de générer les courbes ROC");
            return;
        }
        
        testIterator.reset();
        int numClasses = 0;
        
        // Déterminer le nombre de classes
        while (testIterator.hasNext()) {
            DataSet ds = testIterator.next();
            numClasses = (int) ds.getLabels().size(1);
            break;
        }
        
        if (numClasses == 0) {
            logger.warn("Impossible de déterminer le nombre de classes");
            return;
        }
        
        testIterator.reset();
        
        // Classification binaire ou multi-classe
        if (numClasses == 2) {
            // Courbe ROC pour classification binaire
            ROC roc = new ROC(100);
            
            if (model instanceof MultiLayerNetwork) {
                MultiLayerNetwork net = (MultiLayerNetwork) model;
                net.doEvaluation(testIterator, roc);
            } else {
                logger.warn("Type de modèle non supporté pour ROC: {}", model.getClass().getName());
                return;
            }
            
            // Exporter les données ROC
            String timestamp = dateFormat.format(new Date());
            File outputFile = new File(metricsDir, modelName + "_roc_" + timestamp + ".csv");
            
            try (FileWriter writer = new FileWriter(outputFile)) {
                // En-tête
                writer.write("Threshold,TPR,FPR,Precision,Recall,F1,Accuracy\n");
                
                // Données
                double[] thresholds = roc.getThresholds();
                for (int i = 0; i < thresholds.length; i++) {
                    double threshold = thresholds[i];
                    double tpr = roc.calculateTPR(threshold);
                    double fpr = roc.calculateFPR(threshold);
                    double precision = roc.calculatePrecision(threshold);
                    double recall = roc.calculateRecall(threshold);
                    double f1 = 2 * (precision * recall) / (precision + recall);
                    double accuracy = roc.calculateAccuracy(threshold);
                    
                    writer.write(String.format("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                                            threshold, tpr, fpr, precision, recall, f1, accuracy));
                }
                
                logger.info("Données ROC exportées vers {}", outputFile.getAbsolutePath());
            } catch (IOException e) {
                logger.error("Erreur lors de l'exportation des données ROC : {}", e.getMessage());
            }
            
            // Trouver le seuil optimal (meilleur F1-score)
            double bestThreshold = 0.5;
            double bestF1 = 0.0;
            double[] thresholds = roc.getThresholds();
            
            for (double threshold : thresholds) {
                double precision = roc.calculatePrecision(threshold);
                double recall = roc.calculateRecall(threshold);
                
                if (precision > 0 && recall > 0) {
                    double f1 = 2 * (precision * recall) / (precision + recall);
                    if (f1 > bestF1) {
                        bestF1 = f1;
                        bestThreshold = threshold;
                    }
                }
            }
            
            // Exporter le seuil optimal
            exportOptimalThresholds(Map.of(0, bestThreshold));
            
        } else if (numClasses > 2) {
            // Courbe ROC pour classification multi-classe
            ROCMultiClass rocMultiClass = new ROCMultiClass(100);
            
            if (model instanceof MultiLayerNetwork) {
                MultiLayerNetwork net = (MultiLayerNetwork) model;
                net.doEvaluation(testIterator, rocMultiClass);
            } else {
                logger.warn("Type de modèle non supporté pour ROC multi-classe: {}", model.getClass().getName());
                return;
            }
            
            // Trouver les seuils optimaux pour chaque classe
            Map<Integer, Double> optimalThresholds = new HashMap<>();
            
            for (int i = 0; i < numClasses; i++) {
                double bestThreshold = 0.5;
                double bestF1 = 0.0;
                
                double[] thresholds = rocMultiClass.getThresholds();
                for (double threshold : thresholds) {
                    double precision = rocMultiClass.calculatePrecision(i, threshold);
                    double recall = rocMultiClass.calculateRecall(i, threshold);
                    
                    if (precision > 0 && recall > 0) {
                        double f1 = 2 * (precision * recall) / (precision + recall);
                        if (f1 > bestF1) {
                            bestF1 = f1;
                            bestThreshold = threshold;
                        }
                    }
                }
                
                optimalThresholds.put(i, bestThreshold);
            }
            
            // Exporter les seuils optimaux
            exportOptimalThresholds(optimalThresholds);
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