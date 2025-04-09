package com.project.models;

import com.project.models.activity.ActivityModel;
import com.project.models.activity.ResNetActivityModel;
import com.project.models.activity.VGG16ActivityModel;
import com.project.models.presence.PresenceModel;
import com.project.models.presence.YOLOPresenceModel;
import com.project.models.sound.SoundModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Gestionnaire centralisé pour tous les modèles du projet.
 * Permet de charger, initialiser et utiliser les différents modèles de détection.
 */
public class ModelManager {
    private static final Logger log = LoggerFactory.getLogger(ModelManager.class);
    
    // Types de modèles de présence
    public enum PresenceModelType {
        STANDARD,  // Modèle standard
        YOLO       // Modèle YOLO
    }
    
    // Types de modèles d'activité
    public enum ActivityModelType {
        STANDARD,  // Modèle standard (MobileNetV2)
        VGG16,     // Modèle VGG16
        RESNET     // Modèle ResNet
    }
    
    private final Properties config;
    
    // Modèles de présence
    private PresenceModel standardPresenceModel;
    private YOLOPresenceModel yoloPresenceModel;
    private PresenceModelType currentPresenceModelType;
    
    // Modèles d'activité
    private ActivityModel standardActivityModel;
    private VGG16ActivityModel vgg16ActivityModel;
    private ResNetActivityModel resNetActivityModel;
    private ActivityModelType currentActivityModelType;
    
    // Modèle de son
    private SoundModel soundModel;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ModelManager(Properties config) {
        this.config = config;
        this.currentPresenceModelType = PresenceModelType.valueOf(
                config.getProperty("presence.model.type", "STANDARD").toUpperCase());
        this.currentActivityModelType = ActivityModelType.valueOf(
                config.getProperty("activity.model.type", "STANDARD").toUpperCase());
    }
    
    /**
     * Initialise tous les modèles configurés dans la configuration.
     *
     * @throws IOException Si une erreur survient lors de l'initialisation des modèles
     */
    public void initializeModels() throws IOException {
        // Initialiser le modèle de présence selon la configuration
        initializePresenceModel();
        
        // Initialiser le modèle d'activité selon la configuration
        initializeActivityModel();
        
        // Initialiser le modèle de son
        initializeSoundModel();
    }
    
    /**
     * Initialise le modèle de présence selon le type configuré.
     *
     * @throws IOException Si une erreur survient lors de l'initialisation
     */
    private void initializePresenceModel() throws IOException {
        log.info("Initialisation du modèle de présence de type {}", currentPresenceModelType);
        
        switch (currentPresenceModelType) {
            case YOLO:
                yoloPresenceModel = new YOLOPresenceModel(config);
                yoloPresenceModel.loadDefaultModel();
                break;
            case STANDARD:
            default:
                standardPresenceModel = new PresenceModel(config);
                standardPresenceModel.loadDefaultModel();
                break;
        }
    }
    
    /**
     * Initialise le modèle d'activité selon le type configuré.
     *
     * @throws IOException Si une erreur survient lors de l'initialisation
     */
    private void initializeActivityModel() throws IOException {
        log.info("Initialisation du modèle d'activité de type {}", currentActivityModelType);
        
        switch (currentActivityModelType) {
            case VGG16:
                vgg16ActivityModel = new VGG16ActivityModel(config);
                vgg16ActivityModel.loadDefaultModel();
                break;
            case RESNET:
                resNetActivityModel = new ResNetActivityModel(config);
                resNetActivityModel.loadDefaultModel();
                break;
            case STANDARD:
            default:
                standardActivityModel = new ActivityModel(config);
                standardActivityModel.loadDefaultModel();
                break;
        }
    }
    
    /**
     * Initialise le modèle de son.
     *
     * @throws IOException Si une erreur survient lors de l'initialisation
     */
    private void initializeSoundModel() throws IOException {
        log.info("Initialisation du modèle de son");
        
        soundModel = new SoundModel(config);
        soundModel.loadDefaultModel();
    }
    
    /**
     * Change le type de modèle de présence utilisé.
     *
     * @param modelType Nouveau type de modèle à utiliser
     * @throws IOException Si une erreur survient lors du changement de modèle
     */
    public void setPresenceModelType(PresenceModelType modelType) throws IOException {
        if (this.currentPresenceModelType != modelType) {
            this.currentPresenceModelType = modelType;
            initializePresenceModel();
        }
    }
    
    /**
     * Change le type de modèle d'activité utilisé.
     *
     * @param modelType Nouveau type de modèle à utiliser
     * @throws IOException Si une erreur survient lors du changement de modèle
     */
    public void setActivityModelType(ActivityModelType modelType) throws IOException {
        if (this.currentActivityModelType != modelType) {
            this.currentActivityModelType = modelType;
            initializeActivityModel();
        }
    }
    
    /**
     * Obtient le modèle de présence standard.
     *
     * @return Le modèle de présence standard ou null s'il n'est pas initialisé
     */
    public PresenceModel getStandardPresenceModel() {
        return standardPresenceModel;
    }
    
    /**
     * Obtient le modèle de présence YOLO.
     *
     * @return Le modèle de présence YOLO ou null s'il n'est pas initialisé
     */
    public YOLOPresenceModel getYoloPresenceModel() {
        return yoloPresenceModel;
    }
    
    /**
     * Obtient le modèle d'activité standard.
     *
     * @return Le modèle d'activité standard ou null s'il n'est pas initialisé
     */
    public ActivityModel getStandardActivityModel() {
        return standardActivityModel;
    }
    
    /**
     * Obtient le modèle d'activité VGG16.
     *
     * @return Le modèle d'activité VGG16 ou null s'il n'est pas initialisé
     */
    public VGG16ActivityModel getVgg16ActivityModel() {
        return vgg16ActivityModel;
    }
    
    /**
     * Obtient le modèle d'activité ResNet.
     *
     * @return Le modèle d'activité ResNet ou null s'il n'est pas initialisé
     */
    public ResNetActivityModel getResNetActivityModel() {
        return resNetActivityModel;
    }
    
    /**
     * Obtient le modèle de son.
     *
     * @return Le modèle de son ou null s'il n'est pas initialisé
     */
    public SoundModel getSoundModel() {
        return soundModel;
    }
    
    /**
     * Obtient le type de modèle de présence actuellement utilisé.
     *
     * @return Le type de modèle de présence actuel
     */
    public PresenceModelType getCurrentPresenceModelType() {
        return currentPresenceModelType;
    }
    
    /**
     * Obtient le type de modèle d'activité actuellement utilisé.
     *
     * @return Le type de modèle d'activité actuel
     */
    public ActivityModelType getCurrentActivityModelType() {
        return currentActivityModelType;
    }
}
