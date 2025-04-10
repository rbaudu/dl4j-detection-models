package com.project.common.utils;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class MetricsTracker {
    private static final Logger logger = LoggerFactory.getLogger(MetricsTracker.class);
    
    private String modelName;
    private File metricsDir;
    private List<EvaluationMetrics> metrics;
    private SimpleDateFormat dateFormat;
    
    public MetricsTracker(String modelName, String metricsDir) {
        this.modelName = modelName;
        this.metricsDir = new File(metricsDir);
        if (!this.metricsDir.exists()) {
            this.metricsDir.mkdirs();
        }
        this.metrics = new ArrayList<>();
        this.dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
    }
    
    public void recordEpoch(Model model, DataSetIterator testIterator, int epoch, long trainingTimeMs) {
        if (testIterator == null) {
            logger.warn("Validation iterator is null, skipping evaluation");
            return;
        }
        
        testIterator.reset();
        Evaluation eval = new Evaluation();
        
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) model;
            net.evaluate(testIterator, eval);
        } else {
            logger.warn("Model type not supported for evaluation: {}", model.getClass().getName());
            return;
        }
        
        double accuracy = eval.accuracy();
        double precision = eval.precision();
        double recall = eval.recall();
        double f1 = eval.f1();
        
        EvaluationMetrics epochMetrics = new EvaluationMetrics(accuracy, precision, recall, f1, trainingTimeMs);
        
        // Ajouter les métriques par classe
        for (int i = 0; i < eval.numClasses(); i++) {
            String className = "Class-" + i;
            double classPrecision = eval.precision(i);
            double classRecall = eval.recall(i);
            double classF1 = eval.f1(i);
            
            ClassMetrics classMetrics = new ClassMetrics(i, className, classPrecision, classRecall, classF1);
            classMetrics.setTruePositives((int) eval.truePositives().getDouble(i));
            classMetrics.setFalsePositives((int) eval.falsePositives().getDouble(i));
            classMetrics.setFalseNegatives((int) eval.falseNegatives().getDouble(i));
            
            epochMetrics.addClassMetrics(i, classMetrics);
        }
        
        metrics.add(epochMetrics);
        
        logger.info("Époque {} - {}", epoch, epochMetrics);
        
        exportMetricsToCSV();
        exportClassMetricsToCSV();
    }
    
    public void exportMetricsToCSV() {
        if (metrics.isEmpty()) {
            logger.warn("Aucune métrique à exporter");
            return;
        }
        
        String timestamp = dateFormat.format(new Date());
        File outputFile = new File(metricsDir, modelName + "_metrics_" + timestamp + ".csv");
        
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
            
            logger.info("Métriques exportées vers {}", outputFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Erreur lors de l'exportation des métriques : {}", e.getMessage());
        }
    }
    
    public void exportClassMetricsToCSV() {
        if (metrics.isEmpty() || metrics.get(metrics.size() - 1).getClassMetrics().isEmpty()) {
            return;
        }
        
        EvaluationMetrics latestMetrics = metrics.get(metrics.size() - 1);
        String timestamp = dateFormat.format(new Date());
        File outputFile = new File(metricsDir, modelName + "_class_metrics_" + timestamp + ".csv");
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            // En-tête
            writer.write("Class_Index,Class_Name,Precision,Recall,F1,True_Positives,False_Positives,False_Negatives\n");
            
            // Données
            for (ClassMetrics classMetric : latestMetrics.getClassMetrics().values()) {
                writer.write(String.format("%d,%s,%.4f,%.4f,%.4f,%d,%d,%d\n",
                                         classMetric.getClassIndex(),
                                         classMetric.getClassName() != null ? classMetric.getClassName() : "Classe-" + classMetric.getClassIndex(),
                                         classMetric.getPrecision(),
                                         classMetric.getRecall(),
                                         classMetric.getF1Score(),
                                         classMetric.getTruePositives(),
                                         classMetric.getFalsePositives(),
                                         classMetric.getFalseNegatives()));
            }
            
            logger.info("Métriques par classe exportées vers {}", outputFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Erreur lors de l'exportation des métriques par classe : {}", e.getMessage());
        }
    }
    
    // Getter pour accéder aux métriques collectées
    public List<EvaluationMetrics> getMetrics() {
        return metrics;
    }
}