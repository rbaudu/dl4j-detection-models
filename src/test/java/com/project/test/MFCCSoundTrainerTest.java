package com.project.test;

import com.project.training.MFCCSoundTrainer;
import com.project.training.SpectrogramSoundTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Tests pour les implémentations concrètes des SoundTrainers
 */
public class MFCCSoundTrainerTest {
    private static final Logger log = LoggerFactory.getLogger(MFCCSoundTrainerTest.class);
    
    private Properties config;
    
    @Before
    public void setUp() {
        // Charger la configuration spécifique pour les tests
        config = new Properties();
        try {
            // Essayer d'abord de charger le fichier de test
            File testConfig = new File("config/test-application.properties");
            if (testConfig.exists()) {
                log.info("Chargement de la configuration de test");
                config.load(new FileInputStream(testConfig));
            } else {
                log.info("Configuration de test non trouvée, chargement de la configuration standard");
                config.load(new FileInputStream("config/application.properties"));
            }
            
            // S'assurer que les paramètres nécessaires pour les tests sont définis
            ensureConfigProperty("sound.model.num.classes", "5");
            ensureConfigProperty("sound.input.length", "16000");
            ensureConfigProperty("sound.model.mfcc.coefficients", "40");
            ensureConfigProperty("sound.model.mfcc.length", "300");
            ensureConfigProperty("sound.model.spectrogram.height", "224");
            ensureConfigProperty("sound.model.spectrogram.width", "224");
            ensureConfigProperty("training.epochs", "1");
            ensureConfigProperty("model.hidden.size", "512");
            
        } catch (IOException e) {
            log.warn("Configuration par défaut utilisée", e);
        }
    }
    
    // Helper pour s'assurer qu'une propriété existe
    private void ensureConfigProperty(String key, String defaultValue) {
        if (!config.containsKey(key)) {
            log.info("Ajout de la propriété manquante: {} = {}", key, defaultValue);
            config.setProperty(key, defaultValue);
        }
    }
    
    @Test
    public void testMFCCSoundTrainerInitialization() {
        // Tester la création d'un MFCCSoundTrainer
        MFCCSoundTrainer trainer = new MFCCSoundTrainer(config);
        assertNotNull("Le MFCCSoundTrainer devrait être créé", trainer);
        
        // Vérifier les valeurs des paramètres
        int expectedNumMfcc = Integer.parseInt(config.getProperty("sound.model.mfcc.coefficients", "40"));
        int expectedMfccLength = Integer.parseInt(config.getProperty("sound.model.mfcc.length", "300"));
        long expectedInputSize = expectedNumMfcc * expectedMfccLength;
        
        log.info("Test: MFCCSoundTrainer - Paramètres attendus:");
        log.info("numMfcc = {}", expectedNumMfcc);
        log.info("mfccLength = {}", expectedMfccLength);
        log.info("inputSize (calculé) = {}", expectedInputSize);
        
        // Vérifier que le type est correct
        assertEquals("Le type d'entraîneur devrait être MFCC", 
                     MFCCSoundTrainer.SoundTrainerType.MFCC, trainer.getTrainerType());
    }
    
    @Test
    public void testMFCCSoundTrainerModelCreation() {
        // Créer et initialiser un MFCCSoundTrainer
        MFCCSoundTrainer trainer = new MFCCSoundTrainer(config);
        
        // Initialiser le modèle
        trainer.initializeModel();
        
        // Vérifier que le modèle est créé
        MultiLayerNetwork model = trainer.getModel();
        assertNotNull("Le modèle devrait être créé", model);
        
        // Obtenir la taille d'entrée réelle du modèle
        int actualInputSize = model.getLayer(0).getParam("W").columns();
        
        // Calculer la taille d'entrée attendue
        int expectedNumMfcc = Integer.parseInt(config.getProperty("sound.model.mfcc.coefficients", "40"));
        int expectedMfccLength = Integer.parseInt(config.getProperty("sound.model.mfcc.length", "300"));
        long expectedInputSize = expectedNumMfcc * expectedMfccLength;
        
        log.info("Taille d'entrée réelle du modèle: {}", actualInputSize);
        log.info("Taille d'entrée attendue: {}", expectedInputSize);
        
        // Vérifier que la taille d'entrée correspond aux attentes
        assertEquals("La taille d'entrée du modèle devrait correspondre aux paramètres", 
                (int)expectedInputSize, actualInputSize);
    }
    
    @Test
    public void testSpectrogramSoundTrainerInitialization() {
        // Configurer pour l'approche spectrogramme
        config.setProperty("sound.model.type", "SPECTROGRAM");
        config.setProperty("sound.model.architecture", "VGG16");
        
        // Tester la création d'un SpectrogramSoundTrainer
        SpectrogramSoundTrainer trainer = new SpectrogramSoundTrainer(config);
        assertNotNull("Le SpectrogramSoundTrainer devrait être créé", trainer);
        
        // Vérifier que le type est correct en fonction de l'architecture
        assertEquals("Le type d'entraîneur devrait être SPECTROGRAM_VGG16", 
                     SpectrogramSoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16, trainer.getTrainerType());
    }
    
    @Test
    public void testSpectrogramSoundTrainerModelCreation() {
        // Configurer pour l'approche spectrogramme
        config.setProperty("sound.model.type", "SPECTROGRAM");
        config.setProperty("sound.model.architecture", "STANDARD");
        
        // Créer et initialiser un SpectrogramSoundTrainer
        SpectrogramSoundTrainer trainer = new SpectrogramSoundTrainer(config);
        
        // Initialiser le modèle
        trainer.initializeModel();
        
        // Vérifier que le modèle est créé
        MultiLayerNetwork model = trainer.getModel();
        assertNotNull("Le modèle devrait être créé", model);
        
        // Vérifier que c'est bien un réseau convolutif
        assertTrue("Le modèle devrait être un réseau convolutif", 
                model.getLayer(0).getClass().getSimpleName().toLowerCase().contains("convolution"));
    }
    
    @Test
    public void testSpectrogramVGG16ModelCreation() {
        // Configurer pour l'approche spectrogramme avec VGG16
        config.setProperty("sound.model.type", "SPECTROGRAM");
        config.setProperty("sound.model.architecture", "VGG16");
        
        // Créer et initialiser un SpectrogramSoundTrainer avec VGG16
        SpectrogramSoundTrainer trainer = new SpectrogramSoundTrainer(config);
        
        // Initialiser le modèle
        trainer.initializeModel();
        
        // Vérifier que le modèle est créé
        MultiLayerNetwork model = trainer.getModel();
        assertNotNull("Le modèle VGG16 devrait être créé", model);
        
        // Vérifier que le modèle a plus de couches qu'un réseau standard
        assertTrue("Le modèle VGG16 devrait avoir au moins 8 couches", 
                model.getLayers().length >= 8);
    }
}