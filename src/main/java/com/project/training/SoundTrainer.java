package com.project.training;

import com.project.common.utils.LoggingUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.Random;

/**
 * Classe abstraite de base pour l'entraînement des modèles de sons
 */
public abstract class SoundTrainer {
    private static final Logger logger = LoggerFactory.getLogger(SoundTrainer.class);
    
    // Types d'entraîneurs disponibles
    public enum SoundTrainerType {
        MFCC,
        SPECTROGRAM,
        SPECTROGRAM_VGG16,
        SPECTROGRAM_RESNET
    }
    
    protected Properties config;
    protected int numClasses;
    protected int batchSize;
    protected int numEpochs;
    protected double learningRate;
    protected int seed;
    protected int hiddenLayerSize;
    protected String modelType;
    protected String dataDir;
    protected String modelOutputPath;
    protected MultiLayerNetwork model;
    protected SoundTrainerType trainerType;
    
    /**
     * Constructeur avec configuration
     */
    public SoundTrainer(Properties config) {
        this.config = config;
        
        // Charger les paramètres de base
        this.numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "2"));
        this.batchSize = Integer.parseInt(config.getProperty("training.batch.size", "32"));
        this.numEpochs = Integer.parseInt(config.getProperty("training.epochs", "100"));
        this.learningRate = Double.parseDouble(config.getProperty("training.learning.rate", "0.0001"));
        this.seed = Integer.parseInt(config.getProperty("training.seed", "42"));
        this.hiddenLayerSize = Integer.parseInt(config.getProperty("model.hidden.size", "512"));
        this.modelType = config.getProperty("sound.model.type", "STANDARD");
        this.dataDir = config.getProperty("data.root.dir", "") + "/sound";
        this.modelOutputPath = config.getProperty("models.root.dir", "") + "/sound";
        
        // Définir le type d'entraîneur par défaut
        if ("SPECTROGRAM".equals(modelType)) {
            this.trainerType = SoundTrainerType.SPECTROGRAM_VGG16;
        } else {
            this.trainerType = SoundTrainerType.MFCC;
        }
        
        // Utiliser un générateur de nombres aléatoires fiable
        Random rng = new Random(seed);
        
        logger.info("Initialisation de SoundTrainer avec {} classes", numClasses);
    }
    
    /**
     * Constructeur avec type spécifié
     */
    public SoundTrainer(SoundTrainerType type, Properties config) {
        this(config);
        this.trainerType = type;
        logger.info("Création d'un entraîneur de sons de type: {}", type);
    }
    
    /**
     * Crée un entraîneur de sons selon le type spécifié
     */
    public static SoundTrainer createTrainer(String type, Properties config) {
        logger.info("Création d'un entraîneur de sons de type: {}", type);
        
        if ("MFCC".equals(type)) {
            return new MFCCSoundTrainer(config);
        } else if ("SPECTROGRAM".equals(type)) {
            return new SpectrogramSoundTrainer(config);
        } else if ("SPECTROGRAM_VGG16".equals(type)) {
            return new SpectrogramSoundTrainer(config, "VGG16");
        } else if ("SPECTROGRAM_RESNET".equals(type)) {
            return new SpectrogramSoundTrainer(config, "ResNet");
        } else {
            logger.warn("Type d'entraîneur inconnu: {}, utilisation de MFCC par défaut", type);
            return new MFCCSoundTrainer(config);
        }
    }
    
    /**
     * Méthode pour entraîner le modèle avec une source de données audio
     */
    public MultiLayerNetwork trainOnSoundData(String dataPath, double validationSplit) {
        logger.info("Entraînement du modèle sur les données audio: {}", dataPath);
        // Implémentation spécifique à chaque sous-classe
        return train();
    }
    
    /**
     * Initialise le modèle avec la configuration
     */
    public abstract void initializeModel();
    
    /**
     * Entraîne le modèle avec les données disponibles
     */
    public MultiLayerNetwork train() {
        // Prétraiter les données
        preprocessData();
        
        // Créer le modèle
        model = createModel();
        
        // Afficher les informations sur le modèle et les paramètres d'entraînement
        logTrainingInfo();
        
        // TODO: Implémenter l'entraînement réel du modèle
        
        return model;
    }
    
    /**
     * Récupère le modèle entraîné
     */
    public MultiLayerNetwork getModel() {
        if (model == null) {
            logger.warn("Le modèle n'a pas encore été entraîné");
        }
        return model;
    }
    
    /**
     * Affiche les informations d'entraînement
     */
    protected void logTrainingInfo() {
        LoggingUtils.logSeparator();
        logger.info("=== Paramètres d'entraînement du modèle de son ===");
        logger.info("Type de modèle: {}", modelType);
        logger.info("Nombre de classes: {}", numClasses);
        logger.info("Taille du lot: {}", batchSize);
        logger.info("Nombre d'époques: {}", numEpochs);
        logger.info("Taux d'apprentissage: {}", learningRate);
        logger.info("Répertoire de données: {}", dataDir);
        logger.info("Chemin du modèle: {}", modelOutputPath);
        logger.info("=== Fin des paramètres d'entraînement ===");
        LoggingUtils.logSeparator();
    }
    
    /**
     * Retourne le type d'entraîneur
     */
    public SoundTrainerType getTrainerType() {
        return trainerType;
    }
    
    /**
     * Définit le type d'entraîneur
     */
    public void setTrainerType(SoundTrainerType trainerType) {
        this.trainerType = trainerType;
    }
    
    /**
     * Retourne le nombre de classes pour la classification
     */
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Crée le modèle à entraîner
     */
    protected abstract MultiLayerNetwork createModel();
    
    /**
     * Prétraite les données d'entraînement
     */
    protected abstract void preprocessData();
}