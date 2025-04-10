package com.project.common.utils;

import com.project.common.utils.EvaluationMetrics.ClassMetrics;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Properties;

/**
 * Classe pour suivre et stocker l'évolution des métriques d'évaluation au cours de l'entraînement.
 * Cette classe implémente BaseTrainingListener pour s'intégrer au processus d'entraînement de DL4J.
 */
public class MetricsTracker extends BaseTrainingListener {
    private static final Logger log = LoggerFactory.getLogger(MetricsTracker.class);
    
    private List<EvaluationMetrics> metrics;
    private DataSetIterator validationIterator;
    private int evaluationFrequency;
    private long trainingStartTime;
    private String outputDir;
    private String modelName;
    private boolean exportToTensorBoard;
    private int currentEpoch;
    
    /**
     * Constructeur de MetricsTracker
     * 
     * @param validationIterator Itérateur sur les données de validation
     * @param evaluationFrequency Fréquence d'évaluation (en nombre d'époques)
     * @param outputDir Répertoire de sortie pour les exportations
     * @param modelName Nom du modèle pour l'identification des fichiers
     */
    public MetricsTracker(DataSetIterator validationIterator, int evaluationFrequency,
                          String outputDir, String modelName) {
        this(validationIterator, evaluationFrequency, outputDir, modelName, false);
    }
    
    /**
     * Constructeur de MetricsTracker avec option TensorBoard
     * 
     * @param validationIterator Itérateur sur les données de validation
     * @param evaluationFrequency Fréquence d'évaluation (en nombre d'époques)
     * @param outputDir Répertoire de sortie pour les exportations
     * @param modelName Nom du modèle pour l'identification des fichiers
     * @param exportToTensorBoard Activer l'exportation vers TensorBoard
     */
    public MetricsTracker(DataSetIterator validationIterator, int evaluationFrequency,
                          String outputDir, String modelName, boolean exportToTensorBoard) {
        this.metrics = new ArrayList<>();
        this.validationIterator = validationIterator;
        this.evaluationFrequency = evaluationFrequency > 0 ? evaluationFrequency : 1;
        this.trainingStartTime = System.currentTimeMillis();
        this.outputDir = outputDir;
        this.modelName = modelName;
        this.exportToTensorBoard = exportToTensorBoard;
        this.currentEpoch = 0;
        
        // Créer le répertoire de sortie s'il n'existe pas
        if (outputDir != null) {
            File dir = new File(outputDir);
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    log.warn("Impossible de créer le répertoire de sortie: {}", outputDir);
                }
            }
        }
        
        // Initialiser TensorBoard si nécessaire
        if (exportToTensorBoard) {
            String tensorBoardDir = outputDir + "/tensorboard";
            TensorBoardExporter.initialize(tensorBoardDir);
        }
    }
    
    /**
     * Crée un tracker de métriques à partir des propriétés de configuration
     * 
     * @param validationIterator Itérateur de validation
     * @param config Propriétés de configuration
     * @param modelName Nom du modèle
     * @return Instance de MetricsTracker configurée
     */
    public static MetricsTracker fromConfig(DataSetIterator validationIterator, Properties config, String modelName) {
        String outputDir = config.getProperty("metrics.output.dir", "output/metrics");
        int evaluationFrequency = Integer.parseInt(config.getProperty("metrics.evaluation.frequency", "1"));
        boolean exportToTensorBoard = Boolean.parseBoolean(config.getProperty("tensorboard.enabled", "true"));
        
        return new MetricsTracker(validationIterator, evaluationFrequency, outputDir, modelName, exportToTensorBoard);
    }
    
    /**
     * Appelé à la fin de chaque époque pendant l'entraînement
     */
    @Override
    public void onEpochEnd(Model model) {
        // Incrémenter l'époque manuellement puisque getEpochCount() n'est pas disponible
        currentEpoch++;
        
        // Évaluer le modèle selon la fréquence spécifiée
        if (currentEpoch % evaluationFrequency == 0) {
            long epochTime = System.currentTimeMillis() - trainingStartTime;
            evaluateModel(model, currentEpoch, epochTime);
            trainingStartTime = System.currentTimeMillis(); // Réinitialiser pour la prochaine époque
            
            // Tentative d'exportation des métriques
            try {
                exportMetricsToCSV();
            } catch (IOException e) {
                log.warn("Échec de l'exportation des métriques: {}", e.getMessage());
            }
            
            // Exporter vers TensorBoard si activé
            if (exportToTensorBoard && !metrics.isEmpty()) {
                TensorBoardExporter.exportMetrics(metrics, modelName);
            }
        }
    }
    
    /**
     * Évalue le modèle sur l'ensemble de validation et stocke les métriques
     * 
     * @param model Modèle à évaluer
     * @param epoch Numéro de l'époque
     * @param trainingTime Temps d'entraînement de l'époque
     */
    private void evaluateModel(Model model, int epoch, long trainingTime) {
        if (validationIterator == null) {
            log.warn("Validation iterator is null, skipping evaluation");
            return;
        }
        
        try {
            // Réinitialiser l'itérateur au début
            validationIterator.reset();
            
            // Évaluer le modèle
            Evaluation evaluation = new Evaluation();
            
            while (validationIterator.hasNext()) {
                org.nd4j.linalg.dataset.DataSet next = validationIterator.next();
                INDArray features = next.getFeatures();
                INDArray labels = next.getLabels();
                
                // Obtenir la sortie du modèle (adaptation pour différents types de modèles)
                INDArray output = getModelOutput(model, features);
                
                if (output != null) {
                    evaluation.eval(labels, output);
                }
            }
            
            // Extraire les métriques
            double accuracy = evaluation.accuracy();
            double precision = evaluation.precision();
            double recall = evaluation.recall();
            double f1 = evaluation.f1();
            
            // Créer un objet EvaluationMetrics pour cette époque
            EvaluationMetrics evalMetrics = new EvaluationMetrics(
                epoch, accuracy, precision, recall, f1, trainingTime
            );
            
            // Ajouter les métriques par classe
            int numClasses = evaluation.getNumRowCounter();
            for (int i = 0; i < numClasses; i++) {
                double classPrecision = evaluation.precision(i);
                double classRecall = evaluation.recall(i);
                double classF1 = evaluation.f1(i);
                
                evalMetrics.addClassMetrics(i, classPrecision, classRecall, classF1);
            }
            
            // Ajouter les métriques à la liste
            metrics.add(evalMetrics);
            
            // Journaliser les métriques
            log.info(evalMetrics.toString());
            
            // Exporter directement vers TensorBoard si activé
            if (exportToTensorBoard) {
                TensorBoardExporter.exportEvaluation(evaluation, epoch, modelName);
            }
            
            // Réinitialiser l'itérateur pour la prochaine utilisation
            validationIterator.reset();
            
        } catch (Exception e) {
            log.error("Erreur lors de l'évaluation du modèle: {}", e.getMessage());
        }
    }
    
    /**
     * Méthode adaptée pour obtenir la sortie du modèle de manière sécurisée
     * Cette méthode est adaptative pour différents types de modèles DL4J
     * 
     * @param model Modèle DL4J à évaluer
     * @param features Caractéristiques d'entrée
     * @return Sortie du modèle ou null en cas d'erreur
     */
    private INDArray getModelOutput(Model model, INDArray features) {
        try {
            // Tenter d'obtenir la sortie via différentes approches
            if (model instanceof org.deeplearning4j.nn.multilayer.MultiLayerNetwork) {
                return ((org.deeplearning4j.nn.multilayer.MultiLayerNetwork) model).output(features);
            } else if (model instanceof org.deeplearning4j.nn.graph.ComputationGraph) {
                return ((org.deeplearning4j.nn.graph.ComputationGraph) model).output(features)[0];
            } else {
                // Méthode générique via réflexion (fallback)
                try {
                    java.lang.reflect.Method outputMethod = model.getClass().getMethod("output", INDArray.class);
                    return (INDArray) outputMethod.invoke(model, features);
                } catch (Exception e) {
                    log.error("Impossible d'invoquer la méthode output() via réflexion: {}", e.getMessage());
                    return null;
                }
            }
        } catch (Exception e) {
            log.error("Erreur lors de l'obtention de la sortie du modèle: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Exporte les métriques au format CSV
     * 
     * @throws IOException Si une erreur se produit lors de l'écriture
     */
    public void exportMetricsToCSV() throws IOException {
        if (outputDir == null || metrics.isEmpty()) {
            return;
        }
        
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s/%s_metrics_%s.csv", outputDir, modelName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Écrire l'en-tête
            writer.write("Epoch,Accuracy,Precision,Recall,F1Score,TrainingTime(ms)\n");
            
            // Écrire chaque ligne de métrique
            for (EvaluationMetrics metric : metrics) {
                writer.write(String.format("%d,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                        metric.getEpoch(), metric.getAccuracy(), metric.getPrecision(),
                        metric.getRecall(), metric.getF1Score(), metric.getTrainingTime()));
            }
        }
        
        log.info("Métriques exportées vers {}", filename);
        
        // Exporter également les métriques par classe pour la dernière époque
        if (!metrics.isEmpty()) {
            EvaluationMetrics lastMetrics = metrics.get(metrics.size() - 1);
            String classFilename = String.format("%s/%s_class_metrics_%s.csv", outputDir, modelName, timestamp);
            
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(classFilename))) {
                // Écrire l'en-tête
                writer.write("Class,Precision,Recall,F1Score\n");
                
                // Écrire les métriques pour chaque classe
                for (Integer classIdx : lastMetrics.getPerClassMetrics().keySet()) {
                    ClassMetrics classMetrics = lastMetrics.getClassMetrics(classIdx);
                    writer.write(String.format("%d,%.6f,%.6f,%.6f\n",
                            classIdx.intValue(), classMetrics.getPrecision(),
                            classMetrics.getRecall(), classMetrics.getF1Score()));
                }
            }
            
            log.info("Métriques par classe exportées vers {}", classFilename);
        }
    }
    
    /**
     * Génère un rapport de progression des métriques
     * 
     * @return Rapport formaté
     */
    public String generateProgressReport() {
        if (metrics.isEmpty()) {
            return "Aucune métrique disponible";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("=== Rapport de progression des métriques ===\n");
        sb.append("Époque\tAccuracy\tPrecision\tRecall\tF1-Score\tTemps(ms)\n");
        
        for (EvaluationMetrics metric : metrics) {
            sb.append(String.format("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\n",
                    metric.getEpoch(), metric.getAccuracy(), metric.getPrecision(),
                    metric.getRecall(), metric.getF1Score(), metric.getTrainingTime()));
        }
        
        return sb.toString();
    }
    
    /**
     * Exporte toutes les métriques vers TensorBoard, même si ce n'était pas activé au départ
     * 
     * @param logDir Répertoire de logs TensorBoard
     * @return true si l'export a réussi, false sinon
     */
    public boolean exportToTensorBoard(String logDir) {
        if (metrics.isEmpty()) {
            log.warn("Aucune métrique à exporter vers TensorBoard");
            return false;
        }
        
        // Initialiser TensorBoard
        if (!TensorBoardExporter.initialize(logDir)) {
            log.error("Échec de l'initialisation de TensorBoard pour l'export");
            return false;
        }
        
        // Exporter les métriques
        return TensorBoardExporter.exportMetrics(metrics, modelName);
    }
    
    /**
     * Exporte toutes les métriques vers TensorBoard en utilisant la configuration
     * 
     * @param config Propriétés de configuration
     * @return true si l'export a réussi, false sinon
     */
    public boolean exportToTensorBoard(Properties config) {
        String logDir = config.getProperty("tensorboard.log.dir", "output/tensorboard");
        boolean enableTensorBoard = Boolean.parseBoolean(config.getProperty("tensorboard.enabled", "true"));
        
        if (!enableTensorBoard) {
            log.info("Export TensorBoard désactivé dans la configuration");
            return false;
        }
        
        return exportToTensorBoard(logDir);
    }
    
    /**
     * Retourne les métriques collectées
     * 
     * @return Liste des métriques d'évaluation
     */
    public List<EvaluationMetrics> getMetrics() {
        return metrics;
    }
    
    /**
     * Retourne la dernière métrique collectée
     * 
     * @return Dernière métrique ou null si aucune métrique n'est disponible
     */
    public EvaluationMetrics getLatestMetrics() {
        if (metrics.isEmpty()) {
            return null;
        }
        return metrics.get(metrics.size() - 1);
    }
    
    /**
     * Vérifie si l'exportation vers TensorBoard est activée
     * 
     * @return true si l'exportation TensorBoard est activée
     */
    public boolean isExportToTensorBoardEnabled() {
        return exportToTensorBoard;
    }
    
    /**
     * Active ou désactive l'exportation vers TensorBoard
     * 
     * @param exportToTensorBoard true pour activer, false pour désactiver
     */
    public void setExportToTensorBoard(boolean exportToTensorBoard) {
        this.exportToTensorBoard = exportToTensorBoard;
    }
}