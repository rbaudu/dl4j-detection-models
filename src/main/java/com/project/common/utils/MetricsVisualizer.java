package com.project.common.utils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * Classe pour générer des visualisations graphiques des métriques d'évaluation.
 * Utilise la bibliothèque JFreeChart pour créer des graphiques.
 */
public class MetricsVisualizer {
    private static final Logger log = LoggerFactory.getLogger(MetricsVisualizer.class);
    
    // Largeur et hauteur par défaut des graphiques
    private static final int DEFAULT_WIDTH = 800;
    private static final int DEFAULT_HEIGHT = 600;
    
    // Couleurs pour les différentes métriques
    private static final Color ACCURACY_COLOR = new Color(41, 128, 185); // Bleu
    private static final Color PRECISION_COLOR = new Color(46, 204, 113); // Vert
    private static final Color RECALL_COLOR = new Color(231, 76, 60);    // Rouge
    private static final Color F1_COLOR = new Color(243, 156, 18);       // Orange
    
    /**
     * Génère un graphique linéaire des métriques d'évaluation au fil des époques
     * 
     * @param metrics Liste des métriques d'évaluation
     * @param outputDir Répertoire de sortie pour les graphiques
     * @param modelName Nom du modèle pour l'identification des fichiers
     * @throws IOException Si une erreur survient lors de la création du graphique
     */
    public static void generateEvolutionChart(List<EvaluationMetrics> metrics, 
                                             String outputDir, String modelName) throws IOException {
        if (metrics == null || metrics.isEmpty()) {
            log.warn("Aucune métrique à visualiser");
            return;
        }
        
        // Créer les séries de données
        XYSeries accuracySeries = new XYSeries("Accuracy");
        XYSeries precisionSeries = new XYSeries("Precision");
        XYSeries recallSeries = new XYSeries("Recall");
        XYSeries f1Series = new XYSeries("F1-Score");
        
        // Ajouter les données
        for (EvaluationMetrics metric : metrics) {
            int epoch = metric.getEpoch();
            accuracySeries.add(epoch, metric.getAccuracy());
            precisionSeries.add(epoch, metric.getPrecision());
            recallSeries.add(epoch, metric.getRecall());
            f1Series.add(epoch, metric.getF1Score());
        }
        
        // Créer la collection de séries
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accuracySeries);
        dataset.addSeries(precisionSeries);
        dataset.addSeries(recallSeries);
        dataset.addSeries(f1Series);
        
        // Créer le graphique
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Évolution des métriques - " + modelName,
                "Époque",
                "Valeur",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Personnaliser l'apparence
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        
        // Définir les couleurs et l'épaisseur des lignes
        renderer.setSeriesPaint(0, ACCURACY_COLOR);
        renderer.setSeriesPaint(1, PRECISION_COLOR);
        renderer.setSeriesPaint(2, RECALL_COLOR);
        renderer.setSeriesPaint(3, F1_COLOR);
        
        // Ajouter des points visibles sur les lignes
        for (int i = 0; i < 4; i++) {
            renderer.setSeriesShapesVisible(i, true);
            renderer.setSeriesStroke(i, new BasicStroke(2.0f));
        }
        
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Limiter l'axe Y entre 0 et 1
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setRange(0.0, 1.0);
        
        // Sauvegarder le graphique
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s/%s_metrics_evolution_%s.png", outputDir, modelName, timestamp);
        
        // Vérifier que le répertoire existe
        File dir = new File(outputDir);
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                log.warn("Impossible de créer le répertoire de sortie: {}", outputDir);
            }
        }
        
        // Sauvegarder l'image
        ChartUtils.saveChartAsPNG(new File(filename), chart, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        log.info("Graphique d'évolution des métriques sauvegardé dans {}", filename);
    }
    
    /**
     * Génère un diagramme en barres des dernières métriques par classe
     * 
     * @param metrics Métriques d'évaluation contenant les données par classe
     * @param outputDir Répertoire de sortie pour les graphiques
     * @param modelName Nom du modèle pour l'identification des fichiers
     * @throws IOException Si une erreur survient lors de la création du graphique
     */
    public static void generateClassMetricsChart(EvaluationMetrics metrics, 
                                               String outputDir, String modelName) throws IOException {
        if (metrics == null || metrics.getPerClassMetrics().isEmpty()) {
            log.warn("Aucune métrique par classe à visualiser");
            return;
        }
        
        // Créer le jeu de données
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        // Ajouter les données pour chaque classe
        Map<Integer, EvaluationMetrics.ClassMetrics> classMetricsMap = metrics.getPerClassMetrics();
        for (Map.Entry<Integer, EvaluationMetrics.ClassMetrics> entry : classMetricsMap.entrySet()) {
            int classIndex = entry.getKey();
            EvaluationMetrics.ClassMetrics classMetrics = entry.getValue();
            
            String className = "Classe " + classIndex;
            dataset.addValue(classMetrics.getPrecision(), "Precision", className);
            dataset.addValue(classMetrics.getRecall(), "Recall", className);
            dataset.addValue(classMetrics.getF1Score(), "F1-Score", className);
        }
        
        // Créer le graphique
        JFreeChart chart = ChartFactory.createBarChart(
                "Métriques par classe - " + modelName + " (Époque " + metrics.getEpoch() + ")",
                "Classe",
                "Valeur",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Personnaliser l'apparence
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        
        // Définir les couleurs
        renderer.setSeriesPaint(0, PRECISION_COLOR);
        renderer.setSeriesPaint(1, RECALL_COLOR);
        renderer.setSeriesPaint(2, F1_COLOR);
        
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Limiter l'axe Y entre 0 et 1
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setRange(0.0, 1.0);
        
        // Sauvegarder le graphique
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s/%s_class_metrics_%s.png", outputDir, modelName, timestamp);
        
        // Vérifier que le répertoire existe
        File dir = new File(outputDir);
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                log.warn("Impossible de créer le répertoire de sortie: {}", outputDir);
            }
        }
        
        // Sauvegarder l'image
        ChartUtils.saveChartAsPNG(new File(filename), chart, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        log.info("Graphique des métriques par classe sauvegardé dans {}", filename);
    }
    
    /**
     * Génère un graphique de l'évolution du temps d'entraînement par époque
     * 
     * @param metrics Liste des métriques d'évaluation
     * @param outputDir Répertoire de sortie pour les graphiques
     * @param modelName Nom du modèle pour l'identification des fichiers
     * @throws IOException Si une erreur survient lors de la création du graphique
     */
    public static void generateTrainingTimeChart(List<EvaluationMetrics> metrics, 
                                               String outputDir, String modelName) throws IOException {
        if (metrics == null || metrics.isEmpty()) {
            log.warn("Aucune métrique à visualiser");
            return;
        }
        
        // Créer la série de données
        XYSeries timeSeries = new XYSeries("Temps d'entraînement");
        
        // Ajouter les données
        for (EvaluationMetrics metric : metrics) {
            int epoch = metric.getEpoch();
            double timeInSeconds = metric.getTrainingTime() / 1000.0; // Convertir en secondes
            timeSeries.add(epoch, timeInSeconds);
        }
        
        // Créer la collection de séries
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(timeSeries);
        
        // Créer le graphique
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Temps d'entraînement par époque - " + modelName,
                "Époque",
                "Temps (secondes)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Personnaliser l'apparence
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        
        // Définir la couleur et l'épaisseur de la ligne
        renderer.setSeriesPaint(0, new Color(155, 89, 182)); // Violet
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Sauvegarder le graphique
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s/%s_training_time_%s.png", outputDir, modelName, timestamp);
        
        // Vérifier que le répertoire existe
        File dir = new File(outputDir);
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                log.warn("Impossible de créer le répertoire de sortie: {}", outputDir);
            }
        }
        
        // Sauvegarder l'image
        ChartUtils.saveChartAsPNG(new File(filename), chart, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        log.info("Graphique des temps d'entraînement sauvegardé dans {}", filename);
    }
    
    /**
     * Génère tous les graphiques disponibles pour les métriques fournies
     * 
     * @param metrics Liste des métriques d'évaluation
     * @param outputDir Répertoire de sortie pour les graphiques
     * @param modelName Nom du modèle pour l'identification des fichiers
     * @throws IOException Si une erreur survient lors de la création des graphiques
     */
    public static void generateAllCharts(List<EvaluationMetrics> metrics, 
                                       String outputDir, String modelName) throws IOException {
        if (metrics == null || metrics.isEmpty()) {
            log.warn("Aucune métrique à visualiser");
            return;
        }
        
        // Générer le graphique d'évolution des métriques
        generateEvolutionChart(metrics, outputDir, modelName);
        
        // Générer le graphique des métriques par classe (dernière époque)
        EvaluationMetrics lastMetrics = metrics.get(metrics.size() - 1);
        generateClassMetricsChart(lastMetrics, outputDir, modelName);
        
        // Générer le graphique des temps d'entraînement
        generateTrainingTimeChart(metrics, outputDir, modelName);
        
        log.info("Tous les graphiques ont été générés avec succès");
    }
}
