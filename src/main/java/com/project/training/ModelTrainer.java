package com.project.training;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

import com.project.common.utils.ModelUtils;

/**
 * Classe de base pour l'entraînement des modèles.
 * Cette classe fournit les méthodes communes pour l'entraînement et l'évaluation des modèles.
 */
public abstract class ModelTrainer {

    protected MultiLayerNetwork model;
    protected int batchSize;
    protected int numEpochs;
    protected String modelOutputPath;
    protected Properties config;
    
    /**
     * Constructeur avec paramètres individuels pour les modèles simples
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     */
    public ModelTrainer(int batchSize, int numEpochs, String modelOutputPath) {
        this.batchSize = batchSize;
        this.numEpochs = numEpochs;
        this.modelOutputPath = modelOutputPath;
    }
    
    /**
     * Constructeur avec configuration pour les modèles complexes
     * 
     * @param config Propriétés de configuration
     */
    public ModelTrainer(Properties config) {
        this.config = config;
        this.batchSize = Integer.parseInt(config.getProperty("training.batch.size", "32"));
        this.numEpochs = Integer.parseInt(config.getProperty("training.epochs", "10"));
        this.modelOutputPath = config.getProperty("model.output.path", "models/output/model.zip");
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
            
            // Sauvegarder un checkpoint
            try {
                saveCheckpoint(model, i+1);
            } catch (IOException e) {
                System.err.println("Erreur lors de la sauvegarde du checkpoint : " + e.getMessage());
            }
        }
        
        System.out.println("Entraînement terminé");
        
        // Sauvegarder le modèle final
        saveModel(model);
    }
    
    /**
     * Méthode principale d'entraînement à utiliser par les sous-classes
     * @throws IOException Si une erreur survient
     */
    public void train() throws IOException {
        // Préparer les données
        DataSet data = prepareData();
        if (data == null) {
            throw new IOException("Échec de la préparation des données");
        }
        
        // Obtenir le modèle
        model = getModel();
        if (model == null) {
            throw new IOException("Échec de l'initialisation du modèle");
        }
        
        // Diviser les données en ensembles d'entraînement et de test
        DataSet[] splitData = splitData(data);
        
        // Entraîner le modèle
        train(splitData[0], splitData[1]);
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
            
            // Sauvegarder un checkpoint
            try {
                saveCheckpoint(model, i+1);
            } catch (IOException e) {
                System.err.println("Erreur lors de la sauvegarde du checkpoint : " + e.getMessage());
            }
        }
        
        System.out.println("Entraînement terminé");
        
        // Sauvegarder le modèle final
        saveModel(model);
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
        
        saveModel(model);
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
    
    /**
     * Vérifie si un répertoire existe et le crée si nécessaire
     * @param directory Chemin du répertoire
     * @throws IOException Si une erreur survient lors de la création
     */
    protected void createDirectories(String directory) throws IOException {
        File dir = new File(directory);
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                throw new IOException("Impossible de créer le répertoire : " + directory);
            }
        }
    }
    
    /**
     * Divise les données en ensembles d'entraînement et de test
     * @param data Données complètes
     * @return Tableau avec [données d'entraînement, données de test]
     */
    protected DataSet[] splitData(DataSet data) {
        double trainRatio = Double.parseDouble(config != null ? 
            config.getProperty("training.train.ratio", "0.8") : "0.8");
        
        return splitDataset(data, trainRatio);
    }
    
    /**
     * Divise un dataset en ensembles d'entraînement et de test
     * @param dataset Dataset complet
     * @param trainRatio Ratio pour l'ensemble d'entraînement (0.0-1.0)
     * @return Tableau avec [données d'entraînement, données de test]
     */
    protected DataSet[] splitDataset(DataSet dataset, double trainRatio) {
        int numExamples = dataset.numExamples();
        int trainSize = (int) (numExamples * trainRatio);
        
        dataset.shuffle();
        
        // Utiliser la méthode correcte pour extraire des parties du dataset
        DataSet trainData = dataset.sample(trainSize);
        DataSet testData = dataset.sample(numExamples - trainSize);
        
        return new DataSet[] { trainData, testData };
    }
    
    // Méthodes abstraites à implémenter par les sous-classes
    
    /**
     * Prépare les données pour l'entraînement
     * @return Dataset contenant les données
     * @throws IOException Si une erreur survient
     */
    protected abstract DataSet prepareData() throws IOException;
    
    /**
     * Obtient le modèle à entraîner
     * @return Le modèle initialisé
     */
    protected abstract MultiLayerNetwork getModel();
    
    /**
     * Sauvegarde le modèle entraîné
     * @param network Le modèle à sauvegarder
     * @throws IOException Si une erreur survient
     */
    protected abstract void saveModel(MultiLayerNetwork network) throws IOException;
    
    /**
     * Sauvegarde un checkpoint du modèle
     * @param network Le modèle à sauvegarder
     * @param epoch Numéro de l'époque
     * @throws IOException Si une erreur survient
     */
    protected abstract void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException;
}