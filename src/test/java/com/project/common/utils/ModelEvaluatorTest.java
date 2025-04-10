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
import org.nd4j.evaluation.classification.ROCMultiClass;
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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
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
    private DataSet testData;
    private ModelEvaluator evaluator;
    
    @Before
    public void setUp() throws IOException {
        // Créer un répertoire de sortie temporaire
        File outputDir = tempFolder.newFolder("metrics");
        
        // Préparer une configuration de test
        config = new Properties();
        config.setProperty("metrics.output.dir", outputDir.getAbsolutePath());
        config.setProperty("evaluation.batch.size", "10");
        
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
        
        testData = new DataSet(features, labels);
        
        // Créer l'évaluateur
        evaluator = new ModelEvaluator(model, config);
    }
    
    @Test
    public void testWithLabels() {
        // Tester la méthode withLabels
        List<String> labels = Arrays.asList("Class0", "Class1", "Class2");
        ModelEvaluator labeledEvaluator = evaluator.withLabels(labels);
        
        // Vérifier que la même instance est retournée (pour chaînage)
        assertSame(evaluator, labeledEvaluator);
    }
    
    @Test
    public void testEvaluateAndGenerateReport() throws IOException {
        // Évaluer le modèle
        EvaluationMetrics metrics = evaluator.evaluateAndGenerateReport(testData, "test_model");
        
        // Vérifier que les métriques ne sont pas nulles
        assertNotNull(metrics);
        
        // Vérifier que le rapport a été généré
        File outputDir = new File(config.getProperty("metrics.output.dir"));
        File[] files = outputDir.listFiles((dir, name) -> name.startsWith("test_model_evaluation_report_"));
        
        // Il devrait y avoir au moins un fichier de rapport
        assertTrue(files.length > 0);
        
        // Vérifier le contenu du fichier
        String content = Files.readString(files[0].toPath());
        assertTrue(content.contains("RAPPORT D'ÉVALUATION DU MODÈLE"));
        assertTrue(content.contains("MÉTRIQUES GLOBALES"));
        assertTrue(content.contains("MATRICE DE CONFUSION"));
        assertTrue(content.contains("MÉTRIQUES PAR CLASSE"));
    }
    
    @Test
    public void testMetricsCalculation() throws IOException {
        // Évaluer le modèle et obtenir les métriques
        EvaluationMetrics metrics = evaluator.evaluateAndGenerateReport(testData, "test_metrics");
        
        // Vérifier l'exactitude des métriques
        assertTrue(metrics.getAccuracy() >= 0.0 && metrics.getAccuracy() <= 1.0);
        assertTrue(metrics.getPrecision() >= 0.0 && metrics.getPrecision() <= 1.0);
        assertTrue(metrics.getRecall() >= 0.0 && metrics.getRecall() <= 1.0);
        assertTrue(metrics.getF1Score() >= 0.0 && metrics.getF1Score() <= 1.0);
        assertTrue(metrics.getTrainingTime() >= 0.0);
        
        // Vérifier les métriques par classe
        assertEquals(3, metrics.getPerClassMetrics().size());
        for (int i = 0; i < 3; i++) {
            assertNotNull(metrics.getClassMetrics(i));
        }
    }
    
    @Test
    public void testExportOptimalThresholds() throws IOException {
        // Créer un ROCMultiClass factice
        ROCMultiClass roc = new ROCMultiClass(10);
        
        // Ajouter quelques exemples de test
        for (int i = 0; i < 20; i++) {
            INDArray actual = Nd4j.zeros(1, 3);
            int actualClass = i % 3;
            actual.putScalar(new int[] {0, actualClass}, 1.0);
            
            INDArray predicted = Nd4j.rand(1, 3);
            double sum = predicted.sumNumber().doubleValue();
            predicted.divi(sum);  // Normaliser pour avoir une somme de 1
            
            roc.eval(actual, predicted);
        }
        
        // Exporter les seuils optimaux
        evaluator.exportOptimalThresholds(roc, "test_thresholds");
        
        // Vérifier que le fichier a été créé
        File outputDir = new File(config.getProperty("metrics.output.dir"));
        File[] files = outputDir.listFiles((dir, name) -> name.startsWith("test_thresholds_optimal_thresholds_"));
        
        // Il devrait y avoir un fichier
        assertTrue(files.length > 0);
        
        // Vérifier le contenu du fichier
        String content = Files.readString(files[0].toPath());
        assertTrue(content.startsWith("Class,OptimalThreshold,AUC"));
        
        // Il devrait y avoir une ligne pour chaque classe (+ en-tête)
        assertEquals(4, content.lines().count());
    }
    
    @Test
    public void testValidateMetricsAgainstThresholds() throws IOException {
        // Configurer des seuils
        config.setProperty("test.min.accuracy", "0.7");
        config.setProperty("test.min.precision", "0.7");
        config.setProperty("test.min.recall", "0.7");
        config.setProperty("test.min.f1", "0.7");
        
        // Créer des métriques qui respectent les seuils
        EvaluationMetrics goodMetrics = new EvaluationMetrics(1, 0.8, 0.8, 0.8, 0.8, 100.0);
        
        // Créer des métriques qui ne respectent pas les seuils
        EvaluationMetrics badMetrics = new EvaluationMetrics(1, 0.6, 0.6, 0.6, 0.6, 100.0);
        
        // Tester la validation
        assertTrue(evaluator.validateMetricsAgainstThresholds(goodMetrics));
        assertFalse(evaluator.validateMetricsAgainstThresholds(badMetrics));
    }
    
    @Test
    public void testCalculateMetricsFromConfusionMatrix() {
        // Créer une matrice de confusion fictive
        int[][] confusionMatrix = {
            {50, 5, 5},    // 50 de classe 0 correctement classifiés, 5 classifiés comme classe 1, 5 comme classe 2
            {10, 40, 10},  // 10 de classe 1 classifiés comme classe 0, 40 correctement, 10 comme classe 2
            {5, 5, 70}     // 5 de classe 2 classifiés comme classe 0, 5 comme classe 1, 70 correctement
        };
        
        // Calculer les métriques
        EvaluationMetrics metrics = ModelEvaluator.calculateMetricsFromConfusionMatrix(confusionMatrix);
        
        // Vérifier l'accuracy
        double expectedAccuracy = (50 + 40 + 70) / 200.0;  // 160 corrects sur 200 total
        assertEquals(expectedAccuracy, metrics.getAccuracy(), 0.001);
        
        // Vérifier les métriques par classe
        assertEquals(3, metrics.getPerClassMetrics().size());
        
        // Vérifier la precision pour la classe 0
        double class0Precision = 50.0 / (50 + 10 + 5); // TP / (TP + FP)
        assertEquals(class0Precision, metrics.getClassMetrics(0).getPrecision(), 0.001);
        
        // Vérifier le recall pour la classe 0
        double class0Recall = 50.0 / (50 + 5 + 5); // TP / (TP + FN)
        assertEquals(class0Recall, metrics.getClassMetrics(0).getRecall(), 0.001);
        
        // Vérifier le F1-Score pour la classe 0
        double class0F1 = 2 * class0Precision * class0Recall / (class0Precision + class0Recall);
        assertEquals(class0F1, metrics.getClassMetrics(0).getF1Score(), 0.001);
    }
    
    @Test
    public void testEvaluateWithIterator() throws IOException {
        // Créer un itérateur de données
        DataSetIterator iterator = new ListDataSetIterator<>(Collections.singletonList(testData));
        
        // Évaluer le modèle avec l'itérateur
        EvaluationMetrics metrics = evaluator.evaluateAndGenerateReport(iterator, "iterator_test");
        
        // Vérifier les métriques
        assertNotNull(metrics);
        assertTrue(metrics.getAccuracy() >= 0.0 && metrics.getAccuracy() <= 1.0);
        
        // Vérifier que le rapport a été généré
        File outputDir = new File(config.getProperty("metrics.output.dir"));
        File[] files = outputDir.listFiles((dir, name) -> name.startsWith("iterator_test_evaluation_report_"));
        
        // Il devrait y avoir au moins un fichier
        assertTrue(files.length > 0);
    }
}
