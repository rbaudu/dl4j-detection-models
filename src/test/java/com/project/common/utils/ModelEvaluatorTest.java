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
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe ModelEvaluator
 */
public class ModelEvaluatorTest {
    
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();
    
    private MultiLayerNetwork model;
    private Properties config;
    private DataSetIterator testIterator;
    private ModelEvaluator evaluator;
    private String outputDir;
    
    @Before
    public void setUp() throws IOException {
        // Créer un répertoire de sortie temporaire
        File metricsDir = tempFolder.newFolder("metrics");
        outputDir = metricsDir.getAbsolutePath();
        
        // Préparer une configuration de test
        config = new Properties();
        config.setProperty("metrics.output.dir", outputDir);
        config.setProperty("evaluation.batch.size", "10");
        config.setProperty("test.min.accuracy", "0.7");
        config.setProperty("test.min.precision", "0.7");
        config.setProperty("test.min.recall", "0.7");
        config.setProperty("test.min.f1", "0.7");
        
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
        
        // Créer des données de test simples
        INDArray features = Nd4j.rand(20, 4);
        INDArray labels = Nd4j.zeros(20, 3);
        // Assigner des classes aléatoires
        for (int i = 0; i < 20; i++) {
            int classIdx = (int) (Math.random() * 3);
            labels.putScalar(new int[] {i, classIdx}, 1.0);
        }
        
        DataSet testData = new DataSet(features, labels);
        testIterator = new ListDataSetIterator<>(Collections.singletonList(testData));
        
        // Créer l'évaluateur
        evaluator = new ModelEvaluator("test_model", outputDir);
    }
    
    @Test
    public void testMetricsCalculation() throws IOException {
        // Évaluer le modèle et obtenir les métriques
        EvaluationMetrics metrics = evaluator.evaluate(model, testIterator);
        
        // Vérifier l'exactitude des métriques
        assertTrue(metrics.getAccuracy() >= 0.0 && metrics.getAccuracy() <= 1.0);
        assertTrue(metrics.getPrecision() >= 0.0 && metrics.getPrecision() <= 1.0);
        assertTrue(metrics.getRecall() >= 0.0 && metrics.getRecall() <= 1.0);
        assertTrue(metrics.getF1Score() >= 0.0 && metrics.getF1Score() <= 1.0);
        assertTrue(metrics.getTrainingTime() >= 0);
        
        // Vérifier les métriques par classe
        assertEquals(3, metrics.getClassMetrics().size());
        for (int i = 0; i < 3; i++) {
            assertNotNull(metrics.getClassMetrics().get(i));
        }
    }
    
    @Test
    public void testExportOptimalThresholds() throws IOException {
        // Créer une map de seuils optimaux
        Map<Integer, Double> thresholds = new HashMap<>();
        thresholds.put(0, 0.7);
        thresholds.put(1, 0.65);
        thresholds.put(2, 0.75);
        
        // Exporter les seuils optimaux
        evaluator.exportOptimalThresholds(thresholds);
        
        // Vérifier que le fichier a été créé
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.contains("thresholds") && name.endsWith(".csv"));
        
        // Il devrait y avoir un fichier
        assertNotNull(files);
        assertTrue(files.length > 0);
        
        // Vérifier le contenu du fichier
        String content = Files.readString(files[0].toPath());
        assertTrue(content.contains("ClassIndex,OptimalThreshold"));
        
        // Il devrait y avoir une ligne pour chaque classe (+ en-tête)
        assertEquals(4, content.lines().count());
    }
    
    @Test
    public void testCheckQualityThresholds() {
        // Créer des métriques qui respectent les seuils
        EvaluationMetrics goodMetrics = new EvaluationMetrics(1, 0.8, 0.8, 0.8, 0.8, 100L);
        
        // Créer des métriques qui ne respectent pas les seuils
        EvaluationMetrics badMetrics = new EvaluationMetrics(1, 0.6, 0.6, 0.6, 0.6, 100L);
        
        // Tester la validation
        assertTrue(evaluator.checkQualityThresholds(goodMetrics));
        assertFalse(evaluator.checkQualityThresholds(badMetrics));
    }
    
    @Test
    public void testGenerateConfusionMatrix() {
        // Générer une matrice de confusion pour le modèle
        evaluator.generateConfusionMatrix(model, testIterator);
        
        // Vérifier qu'un fichier a été créé
        File[] files = new File(outputDir).listFiles((dir, name) -> 
            name.contains("confusion") && name.endsWith(".txt"));
        
        // Il devrait y avoir un fichier
        assertNotNull(files);
        assertTrue(files.length > 0);
    }
}