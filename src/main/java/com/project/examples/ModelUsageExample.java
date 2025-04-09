package com.project.examples;

import com.project.common.utils.ModelUtils;
import com.project.config.AppConfig;
import com.project.models.ModelManager;
import com.project.models.activity.ResNetActivityModel;
import com.project.models.activity.VGG16ActivityModel;
import com.project.models.presence.YOLOPresenceModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Exemple d'utilisation des différents modèles.
 * Cette classe démontre comment charger et utiliser les différents modèles de détection.
 */
public class ModelUsageExample {
    private static final Logger log = LoggerFactory.getLogger(ModelUsageExample.class);
    
    public static void main(String[] args) {
        try {
            // Charger la configuration
            Properties config = loadConfiguration("config/application.properties");
            
            // Créer le gestionnaire de modèles
            ModelManager modelManager = new ModelManager(config);
            
            // Initialiser tous les modèles
            modelManager.initializeModels();
            
            // Exemple 1: Utiliser le modèle YOLO pour la détection de présence
            demonstrateYoloPresenceDetection(modelManager);
            
            // Exemple 2: Utiliser le modèle VGG16 pour la détection d'activité
            demonstrateVGG16ActivityDetection(modelManager);
            
            // Exemple 3: Utiliser le modèle ResNet pour la détection d'activité
            demonstrateResNetActivityDetection(modelManager);
            
            // Exemple 4: Passer d'un modèle à l'autre
            demonstrateSwitchingModels(modelManager);
            
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution des exemples", e);
        }
    }
    
    /**
     * Charge le fichier de configuration.
     * 
     * @param configPath Chemin vers le fichier de configuration
     * @return Propriétés de configuration
     * @throws IOException Si une erreur survient lors du chargement de la configuration
     */
    private static Properties loadConfiguration(String configPath) throws IOException {
        Properties config = new Properties();
        File configFile = new File(configPath);
        
        if (!configFile.exists()) {
            throw new IOException("Fichier de configuration non trouvé: " + configPath);
        }
        
        try (InputStream input = new FileInputStream(configFile)) {
            config.load(input);
        }
        
        return config;
    }
    
    /**
     * Démontre l'utilisation du modèle YOLO pour la détection de présence.
     * 
     * @param modelManager Gestionnaire de modèles
     * @throws IOException Si une erreur survient lors du chargement du modèle
     */
    private static void demonstrateYoloPresenceDetection(ModelManager modelManager) throws IOException {
        log.info("=== Exemple d'utilisation du modèle YOLO pour la détection de présence ===");
        
        // Configurer pour utiliser YOLO
        modelManager.setPresenceModelType(ModelManager.PresenceModelType.YOLO);
        
        // Obtenir le modèle YOLO
        YOLOPresenceModel yoloModel = modelManager.getYoloPresenceModel();
        
        // Créer une image de test (simulée)
        INDArray testImage = createTestImage(416, 416);
        
        // Détecter la présence
        boolean presenceDetected = yoloModel.detectPresence(testImage);
        
        log.info("YOLO a détecté une présence: {}", presenceDetected);
    }
    
    /**
     * Démontre l'utilisation du modèle VGG16 pour la détection d'activité.
     * 
     * @param modelManager Gestionnaire de modèles
     * @throws IOException Si une erreur survient lors du chargement du modèle
     */
    private static void demonstrateVGG16ActivityDetection(ModelManager modelManager) throws IOException {
        log.info("=== Exemple d'utilisation du modèle VGG16 pour la détection d'activité ===");
        
        // Configurer pour utiliser VGG16
        modelManager.setActivityModelType(ModelManager.ActivityModelType.VGG16);
        
        // Obtenir le modèle VGG16
        VGG16ActivityModel vgg16Model = modelManager.getVgg16ActivityModel();
        
        // Créer une image de test (simulée)
        INDArray testImage = createTestImage(224, 224);
        
        // Détecter l'activité
        int activityClass = vgg16Model.predictActivity(testImage);
        
        log.info("VGG16 a détecté l'activité de classe: {}", activityClass);
    }
    
    /**
     * Démontre l'utilisation du modèle ResNet pour la détection d'activité.
     * 
     * @param modelManager Gestionnaire de modèles
     * @throws IOException Si une erreur survient lors du chargement du modèle
     */
    private static void demonstrateResNetActivityDetection(ModelManager modelManager) throws IOException {
        log.info("=== Exemple d'utilisation du modèle ResNet pour la détection d'activité ===");
        
        // Configurer pour utiliser ResNet
        modelManager.setActivityModelType(ModelManager.ActivityModelType.RESNET);
        
        // Obtenir le modèle ResNet
        ResNetActivityModel resNetModel = modelManager.getResNetActivityModel();
        
        // Créer une image de test (simulée)
        INDArray testImage = createTestImage(224, 224);
        
        // Détecter l'activité
        int activityClass = resNetModel.predictActivity(testImage);
        
        log.info("ResNet a détecté l'activité de classe: {}", activityClass);
    }
    
    /**
     * Démontre la possibilité de passer facilement d'un modèle à l'autre.
     * 
     * @param modelManager Gestionnaire de modèles
     * @throws IOException Si une erreur survient lors du changement de modèle
     */
    private static void demonstrateSwitchingModels(ModelManager modelManager) throws IOException {
        log.info("=== Exemple de passage d'un modèle à l'autre ===");
        
        // Créer une image de test (simulée)
        INDArray testImagePresence = createTestImage(416, 416);
        INDArray testImageActivity = createTestImage(224, 224);
        
        // Utiliser YOLO pour la détection de présence
        log.info("Utilisation de YOLO pour la détection de présence");
        modelManager.setPresenceModelType(ModelManager.PresenceModelType.YOLO);
        boolean presenceYolo = modelManager.getYoloPresenceModel().detectPresence(testImagePresence);
        
        // Utiliser le modèle standard pour la détection de présence
        log.info("Passage au modèle standard pour la détection de présence");
        modelManager.setPresenceModelType(ModelManager.PresenceModelType.STANDARD);
        // Note: Dans un cas réel, il faudrait implémenter une méthode équivalente pour le modèle standard
        
        // Utiliser VGG16 pour la détection d'activité
        log.info("Utilisation de VGG16 pour la détection d'activité");
        modelManager.setActivityModelType(ModelManager.ActivityModelType.VGG16);
        int activityVGG16 = modelManager.getVgg16ActivityModel().predictActivity(testImageActivity);
        
        // Utiliser ResNet pour la détection d'activité
        log.info("Passage à ResNet pour la détection d'activité");
        modelManager.setActivityModelType(ModelManager.ActivityModelType.RESNET);
        int activityResNet = modelManager.getResNetActivityModel().predictActivity(testImageActivity);
        
        // Utiliser le modèle standard pour la détection d'activité
        log.info("Passage au modèle standard pour la détection d'activité");
        modelManager.setActivityModelType(ModelManager.ActivityModelType.STANDARD);
        // Note: Dans un cas réel, il faudrait implémenter une méthode équivalente pour le modèle standard
        
        log.info("Démonstration de passage entre modèles terminée avec succès");
    }
    
    /**
     * Crée une image de test simulée avec des valeurs aléatoires.
     * 
     * @param height Hauteur de l'image
     * @param width Largeur de l'image
     * @return Image sous forme d'INDArray
     */
    private static INDArray createTestImage(int height, int width) {
        // Créer un tableau 4D: [batchSize, channels, height, width]
        // batchSize = 1, channels = 3 (RGB)
        INDArray image = Nd4j.rand(new int[]{1, 3, height, width});
        
        return image;
    }
}
