package com.project.common.utils;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe MetricsTracker
 */
public class MetricsTrackerTest {
    
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();
    
    private MetricsTracker tracker;
    private MultiLayerNetwork model;
    private DataSetIterator validationIterator;
    private String outputDir;
    
    @Before
    public void setUp() throws IOException {
        // Créer un répertoire de sortie temporaire
        File metricsDir = tempFolder.newFolder("metrics");
        outputDir = metricsDir.getAbsolutePath();
        
        // Créer un modèle simple pour les tests
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(10).nOut(3)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
        
        // Créer des données de validation simples
        INDArray features = Nd4j.rand(20, 4);
        INDArray labels = Nd4j.zeros(20, 3);
        // Assigner des classes aléatoires
        for (int i = 0; i < 20; i++) {
            int classIdx = (int) (Math.random() * 3);
            labels.putScalar(new int[] {i, classIdx}, 1.0);
        }
        
        DataSet validationData = new DataSet(features, labels);
        validationIterator = new ListDataSetIterator<>(Collections.singletonList(validationData));
        
        // Créer le tracker
        tracker = new MetricsTracker(validationIterator, 1, outputDir, "test_model");
    }
    
    @Test
    public void testOnEpochEnd() {
        // Définir le numéro d'époque pour le modèle
        model.setEpochCount(1);
        
        // Appeler onEpochEnd pour déclencher l'évaluation et la collecte des métriques
        tracker.onEpochEnd(model);
        
        // Vérifier que des métriques ont été collectées
        List<EvaluationMetrics> metrics = tracker.getMetrics();
        assertNotNull(metrics);
        assertEquals(1, metrics.size());
        
        // Vérifier les métriques de la première époque
        EvaluationMetrics epoch1Metrics = metrics.get(0);
        assertEquals(1, epoch1Metrics.getEpoch());
        
        // Vérifier que les valeurs des métriques sont dans un intervalle valide
        assertTrue(epoch1Metrics.getAccuracy() >= 0.0 && epoch1Metrics.getAccuracy() <= 1.0);
        assertTrue(epoch1Metrics.getPrecision() >= 0.0 && epoch1Metrics.getPrecision() <= 1.0);
        assertTrue(epoch1Metrics.getRecall() >= 0.0 && epoch1Metrics.getRecall() <= 1.0);
        assertTrue(epoch1Metrics.getF1Score() >= 0.0 && epoch1Metrics.getF1Score() <= 1.0);
        
        // Vérifier que les métriques par classe sont présentes
        assertFalse(epoch1Metrics.getPerClassMetrics().isEmpty());
    }
    
    @Test
    public void testMultipleEpochs() {
        // Simuler plusieurs époques
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.setEpochCount(epoch);
            tracker.onEpochEnd(model);
        }
        
        // Vérifier que les métriques ont été collectées pour chaque époque
        List<EvaluationMetrics> metrics = tracker.getMetrics();
        assertEquals(3, metrics.size());
        
        // Vérifier que les numéros d'époque sont corrects
        assertEquals(1, metrics.get(0).getEpoch());
        assertEquals(2, metrics.get(1).getEpoch());
        assertEquals(3, metrics.get(2).getEpoch());
    }
    
    @Test
    public void testExportMetricsToCSV() throws IOException {
        // Simuler quelques époques
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.setEpochCount(epoch);
            tracker.onEpochEnd(model);
        }
        
        // Exporter les métriques
        tracker.exportMetricsToCSV();
        
        // Vérifier qu'au moins un fichier CSV a été créé
        File outputDirFile = new File(outputDir);
        File[] csvFiles = outputDirFile.listFiles((dir, name) -> name.endsWith(".csv") && name.contains("metrics"));
        
        assertNotNull(csvFiles);
        assertTrue(csvFiles.length > 0);
        
        // Vérifier le contenu du fichier
        String csvContent = Files.readString(csvFiles[0].toPath());
        
        // Vérifier qu'il contient l'en-tête et les données pour 3 époques
        assertTrue(csvContent.contains("Epoch,Accuracy,Precision,Recall,F1Score,TrainingTime"));
        assertEquals(4, csvContent.lines().count()); // 1 ligne d'en-tête + 3 lignes de données
    }
    
    @Test
    public void testGenerateProgressReport() {
        // Simuler quelques époques
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.setEpochCount(epoch);
            tracker.onEpochEnd(model);
        }
        
        // Générer le rapport de progression
        String report = tracker.generateProgressReport();
        
        // Vérifier que le rapport n'est pas vide
        assertNotNull(report);
        assertTrue(report.length() > 0);
        
        // Vérifier qu'il contient les informations attendues
        assertTrue(report.contains("Rapport de progression des métriques"));
        assertTrue(report.contains("Époque\tAccuracy\tPrecision\tRecall\tF1-Score\tTemps(ms)"));
        
        // Il devrait y avoir 3 lignes de données (une par époque) + les en-têtes
        String[] lines = report.split("\n");
        assertTrue(lines.length >= 5); // Au moins 2 lignes d'en-tête + 3 lignes de données
    }
    
    @Test
    public void testGetLatestMetrics() {
        // Pas de métriques au départ
        assertNull(tracker.getLatestMetrics());
        
        // Simuler quelques époques
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.setEpochCount(epoch);
            tracker.onEpochEnd(model);
        }
        
        // Vérifier que getLatestMetrics retourne les métriques de la dernière époque
        EvaluationMetrics latest = tracker.getLatestMetrics();
        assertNotNull(latest);
        assertEquals(3, latest.getEpoch());
    }
    
    @Test
    public void testEvaluationFrequency() {
        // Créer un tracker avec une fréquence d'évaluation de 2 époques
        MetricsTracker trackerWithFrequency = new MetricsTracker(validationIterator, 2, outputDir, "frequency_test");
        
        // Simuler 5 époques
        for (int epoch = 1; epoch <= 5; epoch++) {
            model.setEpochCount(epoch);
            trackerWithFrequency.onEpochEnd(model);
        }
        
        // Vérifier que les métriques ont été collectées seulement pour les époques 2, 4
        List<EvaluationMetrics> metrics = trackerWithFrequency.getMetrics();
        assertEquals(2, metrics.size());
        
        // Vérifier que les numéros d'époque sont corrects
        assertEquals(2, metrics.get(0).getEpoch());
        assertEquals(4, metrics.get(1).getEpoch());
    }
    
    @Test
    public void testNullValidationIterator() {
        // Créer un tracker avec un itérateur null (cas d'erreur)
        MetricsTracker nullTracker = new MetricsTracker(null, 1, outputDir, "null_test");
        
        // Appeler onEpochEnd ne devrait pas provoquer d'exception
        model.setEpochCount(1);
        nullTracker.onEpochEnd(model);
        
        // Vérifier qu'aucune métrique n'a été collectée
        List<EvaluationMetrics> metrics = nullTracker.getMetrics();
        assertTrue(metrics.isEmpty());
    }
}
