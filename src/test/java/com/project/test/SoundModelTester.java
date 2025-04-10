package com.project.test;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Testeur pour les modèles de détection de sons
 * Sert de façade pour les différents types de testeurs spécialisés selon l'approche utilisée
 */
public class SoundModelTester implements ModelTester {
    private static final Logger log = LoggerFactory.getLogger(SoundModelTester.class);
    
    // Types de testeurs de sons
    public enum SoundModelType {
        MFCC,
        SPECTROGRAM
    }
    
    private ModelTester tester;
    private SoundModelType modelType;
    private Properties config;
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public SoundModelTester(Properties config) {
        this.config = config;
        
        // Déterminer le type de testeur
        String modelTypeStr = config.getProperty("sound.model.type", "STANDARD");
        if ("SPECTROGRAM".equalsIgnoreCase(modelTypeStr)) {
            this.modelType = SoundModelType.SPECTROGRAM;
        } else {
            this.modelType = SoundModelType.MFCC;
        }
        
        // Créer le testeur approprié
        createTester();
    }
    
    /**
     * Constructeur avec type de testeur spécifié
     * 
     * @param modelType Type de testeur à utiliser
     * @param config Propriétés de configuration
     */
    public SoundModelTester(SoundModelType modelType, Properties config) {
        this.modelType = modelType;
        this.config = config;
        
        // Créer le testeur approprié
        createTester();
    }
    
    /**
     * Crée le testeur approprié en fonction du type spécifié
     */
    private void createTester() {
        log.info("Création d'un testeur de son de type: {}", modelType);
        
        switch (modelType) {
            case MFCC:
                tester = new MFCCSoundModelTester(config);
                break;
                
            case SPECTROGRAM:
                tester = new SpectrogramSoundModelTester(config);
                break;
                
            default:
                log.warn("Type de testeur inconnu: {}. Utilisation de MFCC par défaut.", modelType);
                tester = new MFCCSoundModelTester(config);
                break;
        }
    }
    
    @Override
    public boolean loadModel() {
        return tester.loadModel();
    }
    
    @Override
    public EvaluationResult testModel() {
        return tester.testModel();
    }
    
    @Override
    public boolean validateModel() {
        return tester.validateModel();
    }
    
    /**
     * Retourne le type de testeur actuel
     * 
     * @return Type de testeur
     */
    public SoundModelType getModelType() {
        return modelType;
    }
    
    /**
     * Change le type de testeur
     * 
     * @param modelType Nouveau type de testeur
     */
    public void setModelType(SoundModelType modelType) {
        if (this.modelType != modelType) {
            this.modelType = modelType;
            createTester();
        }
    }
}
