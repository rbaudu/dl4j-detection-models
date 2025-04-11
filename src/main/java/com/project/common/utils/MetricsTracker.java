package com.project.common.utils;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
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
import java.util.Map;

/**
 * Classe pour suivre les métriques d'entraînement
 */
public class MetricsTracker implements TrainingListener {
    private static final Logger logger = LoggerFactory.getLogger(MetricsTracker.class);
    
    private String modelName;
    private File metricsDir;
    private List<EvaluationMetrics> metrics;
    private SimpleDateFormat dateFormat;
    private DataSetIterator validationIterator;
    private int currentEpoch;
    
    /**
     * Constructeur principal
     */
    public MetricsTracker(String modelName, String metricsDir) {
        this.modelName = modelName;
        this.metricsDir = new File(metricsDir);
        if (!this.metricsDir.exists()) {
            this.metricsDir.mkdirs();
        }
        this.metrics = new ArrayList<>();
        this.dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
        this.currentEpoch = 0;
    }
    
    /**
     * Constructeur étendu pour initialiser avec un iterator de validation
     */
    public MetricsTracker(DataSetIterator validationIterator, int epochs, String modelName, String metricsDir) {
        this(modelName, metricsDir);
        this.validationIterator = validationIterator;
    }
    
    /**
     * Constructeur complet avec paramètre de debug
     */
    public MetricsTracker(DataSetIterator validationIterator, int epochs, String modelName, String metricsDir, boolean debug) {
        this(validationIterator, epochs, modelName, metricsDir);
    }
    
    /**
     * Enregistre les métriques d'une époque
     */
    public void recordEpoch(Model model, DataSetIterator testIterator, int epoch, long trainingTimeMs) {
        if (testIterator == null) {
            logger.warn("Validation iterator is null, skipping evaluation");
            return;
        }
        
        testIterator.reset();
        Evaluation eval = new Evaluation();
        
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) model;
            // Adaptation à l'API réelle
            net.doEvaluation(testIterator, eval);
        } else {
            logger.warn("Model type not supported for evaluation: {}", model.getClass().getName());
            return;
        }
        
        double accuracy = eval.accuracy();
        double precision = eval.precision();
        double recall = eval.recall();
        double f1 = eval.f1();
        
        EvaluationMetrics epochMetrics = new EvaluationMetrics(epoch, accuracy, precision, recall, f1, trainingTimeMs);
        
        // Ajouter les métriques par classe (selon l'API disponible)
        try {
            // Utiliser les labels disponibles pour déterminer les classes
            int classCount = eval.getClasses().size();
            for (int i = 0; i < classCount; i++) {
                String className = "Classe-" + i;
                double classPrecision = eval.precision(i);
                double classRecall = eval.recall(i);
                double classF1 = eval.f1(i);
                
                ClassMetrics classMetrics = new ClassMetrics(i, className, classPrecision, classRecall, classF1);
                
                // Adaptation pour les valeurs de TP/FP/FN
                try {
                    // Utiliser les méthodes disponibles dans l'API DL4J 1.0.0-beta7
                    Map<Integer, Integer> tpMap = eval.truePositives();
                    Map<Integer, Integer> fpMap = eval.falsePositives();
                    Map<Integer, Integer> fnMap = eval.falseNegatives();
                    
                    classMetrics.setTruePositives(tpMap.getOrDefault(i, 0));
                    classMetrics.setFalsePositives(fpMap.getOrDefault(i, 0));
                    classMetrics.setFalseNegatives(fnMap.getOrDefault(i, 0));
                } catch (Exception e) {
                    logger.warn("Impossible de récupérer TP/FP/FN: {}", e.getMessage());
                }
                
                epochMetrics.addClassMetrics(i, classMetrics);
            }
        } catch (Exception e) {
            logger.warn("Erreur lors de l'accès aux classes de l'évaluation: {}", e.getMessage());
        }
        
        metrics.add(epochMetrics);
        
        logger.info("Époque {} - {}", epoch, epochMetrics);
        
        exportMetricsToCSV();
        exportClassMetricsToCSV();
    }
    
    /**
     * Exporte les métriques au format CSV
     */
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
            for (EvaluationMetrics metric : metrics) {
                writer.write(String.format("%d,%.4f,%.4f,%.4f,%.4f,%d\n", 
                                          metric.getEpoch() > 0 ? metric.getEpoch() : metrics.indexOf(metric) + 1, 
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
    
    /**
     * Exporte les métriques par classe au format CSV
     */
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
    
    // Implémentation des méthodes de TrainingListener
    
    @Override
    public void onEpochStart(Model model) {
        currentEpoch++;
        logger.debug("Début de l'époque {}", currentEpoch);
    }
    
    @Override
    public void onEpochEnd(Model model) {
        if (validationIterator != null && model instanceof MultiLayerNetwork) {
            long startTime = System.currentTimeMillis();
            MultiLayerNetwork net = (MultiLayerNetwork) model;
            
            // Faire l'évaluation
            validationIterator.reset();
            Evaluation eval = new Evaluation();
            net.doEvaluation(validationIterator, eval);
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // Enregistrer les métriques
            recordEpoch(model, validationIterator, currentEpoch, duration);
        }
    }
    
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        // Non implémenté
    }
    
    // Méthodes additionnelles requises par l'interface TrainingListener
    
    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        // Non implémenté - requis par l'interface
    }
    
    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        // Non implémenté - requis par l'interface
    }
    
    @Override
    public void onGradientCalculation(Model model) {
        // Non implémenté - requis par l'interface
    }
    
    @Override
    public void onBackwardPass(Model model) {
        // Non implémenté - requis par l'interface
    }
}