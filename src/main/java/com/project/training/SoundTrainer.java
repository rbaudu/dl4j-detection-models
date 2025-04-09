package com.project.training;

import com.project.common.utils.LoggingUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Classe de façade pour les différents types d'entraîneurs de sons.
 * Cette classe délègue les tâches aux implémentations spécialisées.
 */
public class SoundTrainer {
    private static final Logger log = LoggerFactory.getLogger(SoundTrainer.class);
    
    // Types d'entraîneurs de sons
    public enum SoundTrainerType {
        MFCC,
        SPECTROGRAM_VGG16,
        SPECTROGRAM_RESNET
    }
    
    private BaseSoundTrainer trainer;
    private SoundTrainerType trainerType;
    private Properties config;
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public SoundTrainer(Properties config) {
        this.config = config;
        
        // Déterminer le type d'entraîneur
        String modelType = config.getProperty("sound.model.type", "STANDARD");
        if ("SPECTROGRAM".equalsIgnoreCase(modelType)) {
            String architecture = config.getProperty("sound.model.architecture", "VGG16");
            if ("ResNet".equalsIgnoreCase(architecture)) {
                this.trainerType = SoundTrainerType.SPECTROGRAM_RESNET;
            } else {
                this.trainerType = SoundTrainerType.SPECTROGRAM_VGG16;
            }
        } else {
            this.trainerType = SoundTrainerType.MFCC;
        }
        
        // Créer l'entraîneur approprié
        createTrainer();
        
        // Journaliser les paramètres d'entraînement
        LoggingUtils.logSoundTrainingParameters(config);
    }
    
    /**
     * Constructeur avec type d'entraîneur spécifié
     * 
     * @param trainerType Type d'entraîneur à utiliser
     * @param config Propriétés de configuration
     */
    public SoundTrainer(SoundTrainerType trainerType, Properties config) {
        this.trainerType = trainerType;
        this.config = config;
        
        // Créer l'entraîneur approprié
        createTrainer();
    }
    
    /**
     * Crée l'entraîneur approprié en fonction du type spécifié
     */
    private void createTrainer() {
        log.info("Création d'un entraîneur de sons de type: {}", trainerType);
        
        switch (trainerType) {
            case MFCC:
                trainer = new MFCCSoundTrainer(config);
                break;
                
            case SPECTROGRAM_VGG16:
                // Définir l'architecture VGG16
                config.setProperty("sound.model.architecture", "VGG16");
                trainer = new SpectrogramSoundTrainer(config);
                break;
                
            case SPECTROGRAM_RESNET:
                // Définir l'architecture ResNet
                config.setProperty("sound.model.architecture", "ResNet");
                trainer = new SpectrogramSoundTrainer(config);
                break;
                
            default:
                log.warn("Type d'entraîneur inconnu: {}. Utilisation de MFCC par défaut.", trainerType);
                trainer = new MFCCSoundTrainer(config);
                break;
        }
    }
    
    /**
     * Entraîne le modèle sur les données audio
     * 
     * @param dataDir Répertoire contenant les fichiers audio
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnSoundData(String dataDir, double trainTestRatio) throws IOException {
        log.info("Démarrage de l'entraînement de son avec l'entraîneur: {}", trainerType);
        trainer.trainOnSoundData(dataDir, trainTestRatio);
    }
    
    /**
     * Initialise le modèle
     */
    public void initializeModel() {
        trainer.initializeModel();
    }
    
    /**
     * Obtient le modèle entraîné
     * 
     * @return Le modèle entraîné
     */
    public MultiLayerNetwork getModel() {
        return trainer.getModel();
    }
    
    /**
     * Retourne le type d'entraîneur actuel
     * 
     * @return Type d'entraîneur
     */
    public SoundTrainerType getTrainerType() {
        return trainerType;
    }
    
    /**
     * Change le type d'entraîneur
     * 
     * @param trainerType Nouveau type d'entraîneur
     */
    public void setTrainerType(SoundTrainerType trainerType) {
        if (this.trainerType != trainerType) {
            this.trainerType = trainerType;
            createTrainer();
        }
    }
    
    /**
     * Entraîne le modèle avec la configuration par défaut
     * 
     * @throws IOException Si une erreur survient lors de l'entraînement
     */
    public void train() throws IOException {
        String dataDir = config.getProperty("sound.data.dir", "data/raw/sound");
        double trainTestRatio = Double.parseDouble(config.getProperty("training.train.ratio", "0.8"));
        
        trainOnSoundData(dataDir, trainTestRatio);
    }
}
