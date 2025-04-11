package com.project.examples;

import com.project.common.config.ConfigLoader;
import com.project.common.utils.EvaluationMetrics;
import com.project.common.utils.MetricsTracker;
import com.project.common.utils.MetricsVisualizer;
import com.project.common.utils.ModelEvaluator;
import com.project.models.activity.ActivityModel;
import com.project.training.ActivityTrainer;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * Exemple d'utilisation du système de métriques d'évaluation.
 * Cette classe montre comment utiliser les différentes fonctionnalités
 * pour évaluer, suivre et visualiser les performances des modèles.
 */
public class MetricsExampleUsage {
    private static final Logger log = LoggerFactory.getLogger(MetricsExampleUsage.class);
    
    public static void main(String[] args) {
        log.info("Démarrage de l'exemple d'utilisation des métriques d'évaluation");
        
        try {
            // Charger la configuration
            Properties config = ConfigLoader.loadConfiguration();
            
            // Exemple 1: Évaluer un modèle existant
            evaluateExistingModel(config);
            
            // Exemple 2: Suivre les métriques pendant l'entraînement
            trackMetricsDuringTraining(config);
            
            // Exemple 3: Comparer plusieurs modèles
            compareModels(config);
            
            log.info("Exemple d'utilisation des métriques terminé avec succès");
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution de l'exemple", e);
        }
    }
    
    /**
     * Exemple 1: Évaluer un modèle existant et générer un rapport détaillé
     */
    private static void evaluateExistingModel(Properties config) throws IOException {
        log.info("=== Exemple 1: Évaluation d'un modèle existant ===");
        
        // Charger un modèle existant
        ActivityModel model = new ActivityModel(config);
        model.loadDefaultModel();
        
        // Préparer des données de test
        // Note: Dans un cas réel, vous utiliseriez de vraies données de test
        ActivityTrainer trainer = new ActivityTrainer(config);
        DataSet testData = trainer.prepareData("data/raw/activity", 0.2)[1];
        
        // Créer un évaluateur de modèle
        ModelEvaluator evaluator = new ModelEvaluator("activity_model", "output/metrics");
        
        // Convertir le DataSet en DataSetIterator pour l'évaluation
        DataSetIterator testIterator = new ListDataSetIterator<>(
            Collections.singletonList(testData), 32);
        
        // Évaluer le modèle et générer un rapport
        EvaluationMetrics metrics = evaluator.evaluateAndGenerateReport(model.getNetwork(), testIterator);
        
        // Afficher un résumé des métriques
        log.info("Résumé des métriques:");
        log.info("  Accuracy: {}", metrics.getAccuracy());
        log.info("  Precision: {}", metrics.getPrecision());
        log.info("  Recall: {}", metrics.getRecall());
        log.info("  F1-Score: {}", metrics.getF1Score());
        
        // Vérifier si les métriques respectent les seuils minimaux
        boolean valid = evaluator.checkQualityThresholds(metrics);
        log.info("Les métriques respectent les seuils minimaux: {}", valid);
    }
    
    /**
     * Exemple 2: Suivre les métriques pendant l'entraînement
     */
    private static void trackMetricsDuringTraining(Properties config) throws IOException {
        log.info("=== Exemple 2: Suivi des métriques pendant l'entraînement ===");
        
        // Créer un modèle
        ActivityModel model = new ActivityModel(config);
        model.initNewModel();
        
        // Préparer des données d'entraînement et de validation
        // Note: Dans un cas réel, vous utiliseriez de vraies données
        ActivityTrainer trainer = new ActivityTrainer(config);
        DataSet[] datasets = trainer.prepareData("data/raw/activity", 0.8);
        DataSet trainData = datasets[0];
        DataSet validationData = datasets[1];
        
        // Créer des itérateurs pour les données
        DataSetIterator trainIterator = new ListDataSetIterator<>(
            Collections.singletonList(trainData), 32);
        DataSetIterator validationIterator = new ListDataSetIterator<>(
            Collections.singletonList(validationData), 32);
        
        // Créer un tracker de métriques
        String outputDir = "output/metrics";
        MetricsTracker tracker = new MetricsTracker(
            validationIterator, 5, "activity_model", outputDir);
        
        // Ajouter le tracker comme listener au modèle
        if (model.getNetwork() != null) {
            model.getNetwork().setListeners(tracker);
        } else if (model.getGraphNetwork() != null) {
            model.getGraphNetwork().setListeners(tracker);
        }
        
        // Simuler un entraînement (5 époques pour l'exemple)
        log.info("Simulation de l'entraînement avec suivi des métriques...");
        for (int epoch = 0; epoch < 5; epoch++) {
            if (model.getNetwork() != null) {
                model.getNetwork().fit(trainIterator);
            } else if (model.getGraphNetwork() != null) {
                model.getGraphNetwork().fit(trainIterator);
            }
            trainIterator.reset();
            
            // Enregistrer manuellement les métriques pour l'exemple
            if (model.getNetwork() != null) {
                tracker.recordEpoch(model.getNetwork(), validationIterator, epoch+1, 1000);
            } else if (model.getGraphNetwork() != null) {
                tracker.recordEpoch(model.getGraphNetwork(), validationIterator, epoch+1, 1000);
            }
        }
        
        // Générer des graphiques à partir des métriques collectées
        MetricsVisualizer.generateAllCharts(
            tracker.getMetrics(), outputDir, "activity_model");
        
        log.info("Graphiques générés dans {}", outputDir);
    }
    
    /**
     * Exemple 3: Comparer plusieurs modèles
     */
    private static void compareModels(Properties config) throws IOException {
        log.info("=== Exemple 3: Comparaison de plusieurs modèles ===");
        
        // Simuler des métriques pour différents modèles
        // Note: Dans un cas réel, vous utiliseriez de vraies métriques
        
        // Modèle 1: VGG16
        EvaluationMetrics vgg16Metrics = new EvaluationMetrics(
            0.92, 0.89, 0.91, 0.90);
        vgg16Metrics.setEpoch(100);
        vgg16Metrics.setTimeInMs(15000);
        
        // Modèle 2: ResNet
        EvaluationMetrics resnetMetrics = new EvaluationMetrics(
            0.94, 0.92, 0.90, 0.91);
        resnetMetrics.setEpoch(100);
        resnetMetrics.setTimeInMs(18000);
        
        // Modèle 3: MobileNetV2
        EvaluationMetrics mobileNetMetrics = new EvaluationMetrics(
            0.88, 0.87, 0.89, 0.88);
        mobileNetMetrics.setEpoch(100);
        mobileNetMetrics.setTimeInMs(8000);
        
        // Afficher les métriques comparatives
        log.info("Comparaison des modèles:");
        log.info("VGG16: {}", vgg16Metrics);
        log.info("ResNet: {}", resnetMetrics);
        log.info("MobileNet: {}", mobileNetMetrics);
        
        // Dans une application réelle, vous pourriez générer un rapport détaillé ou des visualisations
    }
}