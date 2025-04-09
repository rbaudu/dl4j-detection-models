package com.project.test;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.evaluation.classification.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

/**
 * Classe de base contenant les implémentations communes pour les testeurs de modèles.
 */
public abstract class BaseModelTester {
    private static final Logger log = LoggerFactory.getLogger(BaseModelTester.class);
    
    protected final Properties config;
    protected final String modelType;
    protected MultiLayerNetwork model;
    protected final int inputSize;
    protected final int numClasses;
    protected final int testSamples;
    protected final int batchSize;
    protected final Random random;
    
    public BaseModelTester(Properties config, String modelType) {
        this.config = config;
        this.modelType = modelType;
        
        // Charge les paramètres depuis la configuration
        this.inputSize = Integer.parseInt(config.getProperty(modelType + ".model.input.size", "64"));
        this.numClasses = Integer.parseInt(config.getProperty(modelType + ".model.num.classes", "2"));
        this.testSamples = Integer.parseInt(config.getProperty("test.num.samples", "100"));
        this.batchSize = Integer.parseInt(config.getProperty("test.batch.size", "32"));
        this.random = new Random(Integer.parseInt(config.getProperty("training.seed", "123")));
    }
    
    /**
     * Charge le modèle à tester
     * 
     * @return true si le chargement a réussi
     */
    public boolean loadModel() {
        String modelDir = config.getProperty(modelType + ".model.dir", "models/" + modelType);
        String modelName = config.getProperty(modelType + ".model.name", modelType + "_model");
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        
        try {
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                log.error("Le modèle n'existe pas: {}", modelPath);
                return false;
            }
            
            log.info("Chargement du modèle {}", modelPath);
            model = MultiLayerNetwork.load(modelFile, true);
            log.info("Modèle chargé avec succès");
            return true;
            
        } catch (Exception e) {
            log.error("Erreur lors du chargement du modèle", e);
            return false;
        }
    }
    
    /**
     * Effectue des tests sur le modèle
     * 
     * @return résultat de l'évaluation
     */
    public EvaluationResult testModel() {
        if (model == null) {
            log.error("Le modèle n'est pas chargé, impossible de le tester");
            return null;
        }
        
        // Création de données de test
        DataSet testData = generateTestData();
        if (testData == null) {
            log.error("Échec de génération des données de test");
            return null;
        }
        
        try {
            // Évaluation du modèle
            log.info("Évaluation du modèle...");
            INDArray output = model.output(testData.getFeatures());
            
            // Calculer l'évaluation
            Evaluation eval = new Evaluation(numClasses);
            eval.eval(testData.getLabels(), output);
            
            // Créer le résultat
            EvaluationResult result = new EvaluationResult();
            result.setAccuracy(eval.accuracy());
            result.setPrecision(eval.precision());
            result.setRecall(eval.recall());
            result.setF1Score(eval.f1());
            result.addMetric("Matrice de confusion", eval.getConfusionMatrix().toCSV());
            
            // Afficher les résultats
            log.info("Évaluation terminée:");
            log.info("Précision: {}%", result.getAccuracy() * 100);
            
            return result;
            
        } catch (Exception e) {
            log.error("Erreur lors de l'évaluation du modèle", e);
            return null;
        }
    }
    
    /**
     * Vérifie si le modèle est utilisable par une application externe
     * 
     * @return true si le modèle est validé
     */
    public boolean validateModel() {
        if (model == null) {
            log.error("Le modèle n'est pas chargé, impossible de le valider");
            return false;
        }
        
        try {
            // Vérifier si le modèle peut faire une prédiction simple
            INDArray testInput = generateRandomInput();
            model.output(testInput);
            
            // Vérifier la structure du modèle
            log.info("Structure du modèle: {}", model.summary());
            log.info("Le modèle a passé la validation de base");
            
            return true;
            
        } catch (Exception e) {
            log.error("Le modèle n'a pas passé la validation", e);
            return false;
        }
    }
    
    /**
     * Génère des données de test synthétiques adaptées au modèle
     * 
     * @return DataSet contenant les données de test
     */
    protected DataSet generateTestData() {
        log.info("Génération de données de test synthétiques pour {}", modelType);
        
        // Créer des tableaux pour les entrées et les sorties
        INDArray features = Nd4j.zeros(testSamples, inputSize);
        INDArray labels = Nd4j.zeros(testSamples, numClasses);
        
        // Générer des caractéristiques et des étiquettes
        for (int i = 0; i < testSamples; i++) {
            // Déterminer la classe de cet exemple
            int targetClass = random.nextInt(numClasses);
            
            // Générer des caractéristiques qui favorisent cette classe
            double[] input = generateFeatureVectorForClass(targetClass);
            
            // Ajouter l'exemple au dataset
            for (int j = 0; j < inputSize; j++) {
                features.putScalar(i, j, input[j]);
            }
            
            // Ajouter l'étiquette one-hot
            labels.putScalar(i, targetClass, 1.0);
        }
        
        return new DataSet(features, labels);
    }
    
    /**
     * Génère un vecteur d'entrée aléatoire adapté au modèle
     * 
     * @return INDArray contenant les données d'entrée
     */
    protected INDArray generateRandomInput() {
        double[] input = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            input[i] = random.nextDouble();
        }
        return Nd4j.create(input).reshape(1, inputSize);
    }
    
    /**
     * Génère un vecteur de caractéristiques représentant une classe spécifique
     * Cette méthode doit être implémentée par les sous-classes
     * 
     * @param targetClass Classe cible pour le vecteur
     * @return Tableau de caractéristiques
     */
    protected abstract double[] generateFeatureVectorForClass(int targetClass);
}