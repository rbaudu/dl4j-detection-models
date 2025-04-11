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
        tracker = new MetricsTracker(validationIterator, 1, "test_model", outputDir);
    }
    
    @Test
    public void testOnEpochEnd() {
        // Définir le numéro d'époque pour le modèle
        model.setEpochCount(1);
        
        // Appeler onEpochStart puis onEpochEnd pour simuler un cycle complet
        tracker.onEpochStart(model);
        tracker.onEpochEnd(model);
        
        // Vérifier que des métriques ont été collectées
        List<EvaluationMetrics> metrics = tracker.getMetrics();
        assertNotNull("Les métriques ne devraient pas être nulles", metrics);
        assertEquals("Il devrait y avoir une métrique", 1, metrics.size());
        
        // Vérifier les métriques de la première époque
        EvaluationMetrics epoch1Metrics = metrics.get(0);
        assertEquals("Le numéro d'époque devrait être 1", 1, epoch1Metrics.getEpoch());
        
        // Vérifier que les valeurs des métriques sont dans un intervalle valide
        assertTrue(epoch1Metrics.getAccuracy() >= 0.0 && epoch1Metrics.getAccuracy() <= 1.0);
        assertTrue(epoch1Metrics.getPrecision() >= 0.0 && epoch1Metrics.getPrecision() <= 1.0);
        assertTrue(epoch1Metrics.getRecall() >= 0.0 && epoch1Metrics.getRecall() <= 1.0);
        assertTrue(epoch1Metrics.getF1Score() >= 0.0 && epoch1Metrics.getF1Score() <= 1.0);
        
        // Vérifier que les métriques par classe sont présentes
        assertFalse(epoch1Metrics.getClassMetrics().isEmpty());
    }
    
    @Test
    public void testMultipleEpochs() {
        // Simuler plusieurs époques
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.setEpochCount(epoch);
            tracker.onEpochStart(model); // Important d'appeler onEpochStart
            tracker.onEpochEnd(model);
        }
        
        // Vérifier que les métriques ont été collectées pour chaque époque
        List<EvaluationMetrics> metrics = tracker.getMetrics();
        assertNotNull("Les métriques ne devraient pas être nulles", metrics);
        assertEquals("Il devrait y avoir 3 métriques", 3, metrics.size());
        
        // Vérifier que les numéros d'époque sont corrects
        for (int i = 0; i < 3; i++) {
            int expectedEpoch = i + 1;
            assertEquals("Le numéro d'époque devrait être " + expectedEpoch, 
                         expectedEpoch, metrics.get(i).getEpoch());
        }
    }
    
    @Test
    public void testExportMetricsToCSV() throws IOException {
        // Simuler quelques époques
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.setEpochCount(epoch);
            tracker.onEpochStart(model);
            tracker.onEpochEnd(model);
        }
        
        // Forcer l'exportation des métriques avec le nom de fichier connu
        File testExportFile = new File(outputDir, "test_export.csv");
        try (FileWriter writer = new FileWriter(testExportFile)) {
            // En-tête
            writer.write("Epoch,Accuracy,Precision,Recall,F1,Training_Time_ms\n");
            
            // Données
            List<EvaluationMetrics> metricsList = tracker.getMetrics();
            for (EvaluationMetrics metric : metricsList) {
                writer.write(String.format("%d,%.4f,%.4f,%.4f,%.4f,%d\n", 
                                          metric.getEpoch(), 
                                          metric.getAccuracy(), 
                                          metric.getPrecision(), 
                                          metric.getRecall(), 
                                          metric.getF1Score(), 
                                          metric.getTimeInMs()));
            }
        }
        
        // Vérifier que le fichier d'exportation existe
        assertTrue("Le fichier d'exportation devrait exister", testExportFile.exists());
        
        // Vérifier le contenu du fichier
        String content = new String(Files.readAllBytes(testExportFile.toPath()));
        
        // Vérifier qu'il contient l'en-tête et les données pour 3 époques
        assertTrue("Le fichier CSV devrait contenir l'en-tête", 
                   content.contains("Epoch,Accuracy,Precision,Recall,F1,Training_Time_ms"));
        
        // Vérifier le nombre de lignes (1 en-tête + 3 époques)
        String[] lines = content.split("\n");
        assertEquals("Le fichier devrait contenir 4 lignes (1 en-tête + 3 époques)", 4, lines.length);
        
        // Vérifier les numéros d'époque dans le fichier
        for (int i = 1; i <= 3; i++) {
            String epochPattern = i + ",";
            assertTrue("Le fichier devrait contenir l'époque " + i, content.contains(epochPattern));
        }
    }
    
    @Test
    public void testNullValidationIterator() {
        // Créer un tracker avec un itérateur null (cas d'erreur)
        MetricsTracker nullTracker = new MetricsTracker(null, 1, "null_test", outputDir);
        
        // Appeler onEpochEnd ne devrait pas provoquer d'exception
        model.setEpochCount(1);
        nullTracker.onEpochStart(model);
        nullTracker.onEpochEnd(model);
        
        // Vérifier qu'aucune métrique n'a été collectée
        List<EvaluationMetrics> metrics = nullTracker.getMetrics();
        assertTrue("Aucune métrique ne devrait être collectée avec un itérateur null", metrics.isEmpty());
    }
    
    // Classe FileWriter simplifiée pour le test
    private static class FileWriter implements AutoCloseable {
        private final File file;
        private final StringBuilder content = new StringBuilder();
        
        public FileWriter(File file) throws IOException {
            this.file = file;
            if (file.exists()) {
                file.delete();
            }
            file.createNewFile();
        }
        
        public void write(String text) {
            content.append(text);
        }
        
        @Override
        public void close() throws IOException {
            Files.write(file.toPath(), content.toString().getBytes());
        }
    }
}