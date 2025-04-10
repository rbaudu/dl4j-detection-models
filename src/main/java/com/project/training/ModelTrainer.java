package com.project.training;

import com.project.common.utils.EvaluationMetrics;
import com.project.common.utils.MetricsTracker;
import com.project.common.utils.MetricsVisualizer;
import com.project.common.utils.ModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Classe de base pour l'entraînement des modèles.
 * Cette classe fournit les méthodes communes pour l'entraînement et l'évaluation des modèles.
 */
public abstract class ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(ModelTrainer.class);

    protected MultiLayerNetwork model;
    protected int batchSize;
    protected int numEpochs;
    protected String modelOutputPath;
    protected Properties config;
    protected MetricsTracker metricsTracker;
    
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
        
        // Initialiser le tracker de métriques si ce n'est pas déjà fait
        if (metricsTracker == null) {
            String outputDir = config != null ? 
                config.getProperty("metrics.output.dir", "output/metrics") : 
                "output/metrics";
            
            String modelName = getModelName();
            int evaluationFrequency = 1; // Évaluer à chaque époque
            
            metricsTracker = new MetricsTracker(testIterator, evaluationFrequency, outputDir, modelName);
            
            // Créer le répertoire de sortie s'il n'existe pas
            createDirectories(outputDir);
        }
        
        // Ajouter le tracker de métriques comme listener
        model.setListeners(new ScoreIterationListener(10), metricsTracker);
        
        log.info("Début de l'entraînement pour {} époques...", numEpochs);
        
        for (int i = 0; i < numEpochs; i++) {
            long epochStartTime = System.currentTimeMillis();
            model.fit(trainIterator);
            
            // Évaluer le modèle sur les données de test après chaque époque
            Evaluation evaluation = model.evaluate(testIterator);
            log.info("Époque {} / {} terminée", (i + 1), numEpochs);
            log.info(evaluation.stats());
            
            // Réinitialiser les itérateurs pour la prochaine époque
            trainIterator.reset();
            testIterator.reset();
            
            // Sauvegarder un checkpoint
            try {
                saveCheckpoint(model, i+1);
            } catch (IOException e) {
                log.error("Erreur lors de la sauvegarde du checkpoint : {}", e.getMessage());
            }
        }
        
        log.info("Entraînement terminé");
        
        // Exporter les métriques sous forme de graphiques
        try {
            String outputDir = config != null ? 
                config.getProperty("metrics.output.dir", "output/metrics") : 
                "output/metrics";
                
            String modelName = getModelName();
            
            // Générer les graphiques
            MetricsVisualizer.generateAllCharts(
                metricsTracker.getMetrics(),
                outputDir,
                modelName
            );
            
            log.info("Graphiques de métriques générés avec succès");
        } catch (Exception e) {
            log.error("Erreur lors de la génération des graphiques de métriques : {}", e.getMessage());
        }
        
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
        
        // Initialiser le tracker de métriques
        String outputDir = config != null ? 
            config.getProperty("metrics.output.dir", "output/metrics") : 
            "output/metrics";
        
        String modelName = getModelName();
        int evaluationFrequency = 1; // Évaluer à chaque époque
        
        // Créer le répertoire de sortie s'il n'existe pas
        createDirectories(outputDir);
        
        metricsTracker = new MetricsTracker(testIterator, evaluationFrequency, outputDir, modelName);
        
        // Ajouter le tracker de métriques comme listener
        model.setListeners(new ScoreIterationListener(10), metricsTracker);
        
        log.info("Début de l'entraînement pour {} époques...", numEpochs);
        
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIterator);
            
            // Évaluer le modèle sur les données de test après chaque époque
            Evaluation evaluation = model.evaluate(testIterator);
            log.info("Époque {} / {} terminée", (i + 1), numEpochs);
            log.info(evaluation.stats());
            
            // Réinitialiser les itérateurs pour la prochaine époque
            trainIterator.reset();
            testIterator.reset();
            
            // Sauvegarder un checkpoint
            try {
                saveCheckpoint(model, i+1);
            } catch (IOException e) {
                log.error("Erreur lors de la sauvegarde du checkpoint : {}", e.getMessage());
            }
        }
        
        log.info("Entraînement terminé");
        
        // Exporter les métriques sous forme de graphiques
        try {
            // Générer les graphiques
            MetricsVisualizer.generateAllCharts(
                metricsTracker.getMetrics(),
                outputDir,
                modelName
            );
            
            log.info("Graphiques de métriques générés avec succès");
        } catch (Exception e) {
            log.error("Erreur lors de la génération des graphiques de métriques : {}", e.getMessage());
        }
        
        // Sauvegarder le modèle final
        saveModel(model);
    }
    
    /**
     * Évalue le modèle sur un ensemble de données et retourne les métriques détaillées
     * @param testData Données de test
     * @return Métriques d'évaluation
     */
    public EvaluationMetrics evaluateWithMetrics(DataSet testData) {
        if (model == null) {
            throw new IllegalStateException("Le modèle doit être initialisé avant l'évaluation");
        }
        
        // Créer un itérateur pour les données de test
        DataSetIterator testIterator = new TestDataSetIterator(testData, batchSize);
        
        // Mesurer le temps de départ
        long startTime = System.currentTimeMillis();
        
        // Évaluer le modèle
        Evaluation evaluation = model.evaluate(testIterator);
        
        // Calculer le temps écoulé
        long endTime = System.currentTimeMillis();
        long elapsed = endTime - startTime;
        
        // Extraire les métriques
        double accuracy = evaluation.accuracy();
        double precision = evaluation.precision();
        double recall = evaluation.recall();
        double f1 = evaluation.f1();
        
        // Créer l'objet de métriques
        EvaluationMetrics metrics = new EvaluationMetrics(
            0, accuracy, precision, recall, f1, elapsed
        );
        
        // Ajouter les métriques par classe
        int numClasses = evaluation.getNumRowCounter().rows();
        for (int i = 0; i < numClasses; i++) {
            double classPrecision = evaluation.precision(i);
            double classRecall = evaluation.recall(i);
            double classF1 = evaluation.f1(i);
            
            metrics.addClassMetrics(i, classPrecision, classRecall, classF1);
        }
        
        return metrics;
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
        log.info("Modèle chargé depuis {}", modelPath);
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
    
    /**
     * Obtient le nom du modèle à partir de la configuration ou une valeur par défaut
     * @return Nom du modèle
     */
    protected String getModelName() {
        if (config != null) {
            // Essayer de déterminer le type de modèle à partir de la configuration
            String modelType = "unknown";
            
            if (config.containsKey("presence.model.type")) {
                modelType = "presence_" + config.getProperty("presence.model.type").toLowerCase();
            } else if (config.containsKey("activity.model.type")) {
                modelType = "activity_" + config.getProperty("activity.model.type").toLowerCase();
            } else if (config.containsKey("sound.model.type")) {
                modelType = "sound_" + config.getProperty("sound.model.type").toLowerCase();
            }
            
            return modelType;
        }
        
        return "model";
    }
    
    /**
     * Obtient les métriques collectées
     * 
     * @return Tracker de métriques
     */
    public MetricsTracker getMetricsTracker() {
        return metricsTracker;
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