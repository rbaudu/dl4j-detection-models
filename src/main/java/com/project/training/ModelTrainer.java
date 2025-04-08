package com.project.training;

import com.project.common.utils.DataProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Classe abstraite de base pour l'entraînement des modèles.
 * Définit le flux de travail général pour l'entraînement tout en laissant les détails
 * spécifiques aux sous-classes.
 */
public abstract class ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(ModelTrainer.class);
    
    protected final Properties config;
    protected final int batchSize;
    protected final int numEpochs;
    protected final double trainTestSplit;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ModelTrainer(Properties config) {
        this.config = config;
        this.batchSize = Integer.parseInt(config.getProperty("training.batch.size", "32"));
        this.numEpochs = Integer.parseInt(config.getProperty("training.epochs", "100"));
        this.trainTestSplit = Double.parseDouble(config.getProperty("training.train.test.split", "0.8"));
    }
    
    /**
     * Méthode principale pour lancer l'entraînement.
     * Coordonne le processus d'entraînement complet.
     *
     * @throws IOException en cas d'erreur lors de l'entraînement
     */
    public void train() throws IOException {
        log.info("Démarrage de l'entraînement du modèle avec {} époques", numEpochs);
        
        // Préparer les données
        DataSet dataset = prepareData();
        if (dataset == null || dataset.isEmpty()) {
            throw new IOException("Impossible de préparer les données d'entraînement");
        }
        
        log.info("Données préparées: {} exemples", dataset.numExamples());
        
        // Diviser en ensembles d'entraînement et de test
        SplitTestAndTrain testAndTrain = dataset.splitTestAndTrain(trainTestSplit);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        
        log.info("Division des données: {} exemples d'entraînement, {} exemples de test",
                trainingData.numExamples(), testData.numExamples());
        
        // Obtenir et configurer le modèle
        MultiLayerNetwork model = getModel();
        if (model == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé");
        }
        
        // Ajouter des listeners pour suivre la progression
        model.setListeners(new ScoreIterationListener(10));
        
        // Entraîner le modèle
        log.info("Début de l'entraînement pour {} époques avec une taille de batch de {}", 
                numEpochs, batchSize);
        
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
            
            // Évaluer périodiquement
            if ((i + 1) % 10 == 0 || i == numEpochs - 1) {
                log.info("Époque {} terminée", i + 1);
                evaluateModel(model, testData);
                
                // Sauvegarder le checkpoint
                if ((i + 1) % 20 == 0) {
                    saveCheckpoint(model, i + 1);
                }
            }
        }
        
        log.info("Entraînement terminé");
        
        // Évaluation finale
        log.info("Évaluation finale du modèle");
        evaluateModel(model, testData);
        
        // Sauvegarder le modèle final
        saveModel(model);
        
        log.info("Modèle sauvegardé et prêt à être utilisé");
    }
    
    /**
     * Prépare les données pour l'entraînement.
     * À implémenter par les sous-classes pour charger et prétraiter les données.
     *
     * @return DataSet contenant les données préparées
     * @throws IOException en cas d'erreur lors de la préparation des données
     */
    protected abstract DataSet prepareData() throws IOException;
    
    /**
     * Obtient le modèle à entraîner.
     * À implémenter par les sous-classes pour fournir le modèle spécifique.
     *
     * @return Le modèle neuronal à entraîner
     */
    protected abstract MultiLayerNetwork getModel();
    
    /**
     * Sauvegarde le modèle dans son emplacement par défaut.
     * À implémenter par les sous-classes pour la sauvegarde spécifique au modèle.
     *
     * @param model Le modèle entraîné à sauvegarder
     * @throws IOException en cas d'erreur lors de la sauvegarde
     */
    protected abstract void saveModel(MultiLayerNetwork model) throws IOException;
    
    /**
     * Sauvegarde un checkpoint du modèle pendant l'entraînement.
     *
     * @param model Le modèle à l'état actuel
     * @param epoch Numéro de l'époque actuelle
     * @throws IOException en cas d'erreur lors de la sauvegarde
     */
    protected abstract void saveCheckpoint(MultiLayerNetwork model, int epoch) throws IOException;
    
    /**
     * Évalue le modèle sur les données de test.
     *
     * @param model Le modèle à évaluer
     * @param testData Données de test
     */
    protected void evaluateModel(MultiLayerNetwork model, DataSet testData) {
        Evaluation eval = model.evaluate(testData);
        log.info("Résultats de l'évaluation:\n{}", eval.stats());
    }
    
    /**
     * Crée les répertoires nécessaires s'ils n'existent pas.
     *
     * @param modelDir Répertoire principal du modèle
     * @throws IOException en cas d'erreur lors de la création des répertoires
     */
    protected void createDirectories(String modelDir) throws IOException {
        // Créer le répertoire principal du modèle
        DataProcessor.ensureDirectoryExists(modelDir);
        
        // Créer le répertoire des checkpoints
        DataProcessor.ensureDirectoryExists(modelDir + "/checkpoints");
    }
}
