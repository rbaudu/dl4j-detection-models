package com.project.training;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;

import java.io.IOException;
import java.util.Arrays;

import com.project.common.utils.ModelUtils;

public class ModelTrainer {

    protected MultiLayerNetwork model;
    protected int batchSize;
    protected int numEpochs;
    protected String modelOutputPath;
    
    public ModelTrainer(int batchSize, int numEpochs, String modelOutputPath) {
        this.batchSize = batchSize;
        this.numEpochs = numEpochs;
        this.modelOutputPath = modelOutputPath;
    }
    
    /**
     * Initialise le modèle à utiliser pour l'entraînement
     * @param numInputs Nombre d'entrées
     * @param numOutputs Nombre de sorties (classes)
     */
    public void initializeModel(int numInputs, int numOutputs) {
        model = ModelUtils.createSimpleNetwork(numInputs, numOutputs);
        model.setListeners(new ScoreIterationListener(10));
    }
    
    /**
     * Entraîne le modèle avec les données fournies
     * @param trainData Données d'entraînement
     * @param testData Données de test
     * @throws IOException Si une erreur survient lors de la sauvegarde du modèle
     */
    public void train(DataSet trainData, DataSet testData) throws IOException {
        if (model == null) {
            throw new IllegalStateException("Le modèle doit être initialisé avant l'entraînement");
        }
        
        // Créer des itérateurs pour les données d'entraînement et de test
        DataSetIterator trainIterator = new TestDataSetIterator(trainData, batchSize);
        DataSetIterator testIterator = new TestDataSetIterator(testData, batchSize);
        
        System.out.println("Début de l'entraînement...");
        
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIterator);
            
            // Évaluer le modèle sur les données de test après chaque époque
            Evaluation evaluation = model.evaluate(testIterator);
            System.out.println("Époque " + (i + 1) + "/" + numEpochs);
            System.out.println(evaluation.stats());
            
            // Réinitialiser les itérateurs pour la prochaine époque
            trainIterator.reset();
            testIterator.reset();
        }
        
        System.out.println("Entraînement terminé");
        
        // Sauvegarder le modèle
        saveModel();
    }
    
    /**
     * Entraîne le modèle avec les données fournies
     * @param trainIterator Itérateur pour les données d'entraînement
     * @param testIterator Itérateur pour les données de test
     * @throws IOException Si une erreur survient lors de la sauvegarde du modèle
     */
    public void train(DataSetIterator trainIterator, DataSetIterator testIterator) throws IOException {
        if (model == null) {
            throw new IllegalStateException("Le modèle doit être initialisé avant l'entraînement");
        }
        
        System.out.println("Début de l'entraînement...");
        
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIterator);
            
            // Évaluer le modèle sur les données de test après chaque époque
            Evaluation evaluation = model.evaluate(testIterator);
            System.out.println("Époque " + (i + 1) + "/" + numEpochs);
            System.out.println(evaluation.stats());
            
            // Réinitialiser les itérateurs pour la prochaine époque
            trainIterator.reset();
            testIterator.reset();
        }
        
        System.out.println("Entraînement terminé");
        
        // Sauvegarder le modèle
        saveModel();
    }
    
    /**
     * Évalue le modèle sur un ensemble de données
     * @param testData Données de test
     * @return Les statistiques d'évaluation
     */
    public String evaluate(DataSet testData) {
        if (model == null) {
            throw new IllegalStateException("Le modèle doit être initialisé avant l'évaluation");
        }
        
        // Créer un itérateur pour les données de test
        DataSetIterator testIterator = new TestDataSetIterator(testData, batchSize);
        
        // Évaluer le modèle
        Evaluation evaluation = model.evaluate(testIterator);
        
        return evaluation.stats();
    }
    
    /**
     * Sauvegarde le modèle dans le chemin spécifié
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public void saveModel() throws IOException {
        if (model == null) {
            throw new IllegalStateException("Le modèle doit être initialisé avant la sauvegarde");
        }
        
        ModelUtils.saveModel(model, modelOutputPath, true);
        System.out.println("Modèle sauvegardé à " + modelOutputPath);
    }
    
    /**
     * Charge un modèle existant
     * @param modelPath Chemin du modèle à charger
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        model = ModelUtils.loadModel(modelPath);
        System.out.println("Modèle chargé depuis " + modelPath);
    }
}