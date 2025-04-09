package com.project.config;

/**
 * Classe de configuration centralisée pour l'application
 */
public class AppConfig {
    // Chemins des fichiers et répertoires
    public static final String BASE_MODEL_PATH = "models/base/";
    public static final String TRAINED_MODEL_PATH = "models/trained/";
    public static final String ACTIVITY_MODEL_PATH = TRAINED_MODEL_PATH + "activity_model.zip";
    public static final String SOUND_MODEL_PATH = TRAINED_MODEL_PATH + "sound_model.zip";
    
    // Paramètres d'entraînement
    public static final int BATCH_SIZE = 32;
    public static final int NUM_EPOCHS = 10;
    public static final double DEFAULT_TRAIN_RATIO = 0.8;
    
    // Paramètres pour la détection d'activité
    public static final int ACTIVITY_IMAGE_HEIGHT = 64;
    public static final int ACTIVITY_IMAGE_WIDTH = 64;
    public static final int ACTIVITY_IMAGE_CHANNELS = 3; // RGB
    public static final int ACTIVITY_NUM_CLASSES = 5;
    
    // Paramètres pour la classification de sons
    public static final int SOUND_FEATURE_SIZE = 128;
    public static final int SOUND_NUM_CLASSES = 3;
    
    // Paramètres d'extraction de caractéristiques audio
    public static final int AUDIO_WINDOW_SIZE = 512;
    public static final int AUDIO_HOP_SIZE = 256;
    
    // Paramètres pour les modèles de réseau de neurones
    public static final double LEARNING_RATE = 0.001;
    public static final double L2_REGULARIZATION = 1e-5;
    public static final int RANDOM_SEED = 123;
}