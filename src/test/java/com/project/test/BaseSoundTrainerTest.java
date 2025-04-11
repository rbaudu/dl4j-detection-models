package com.project.test;

import com.project.training.BaseSoundTrainer;
import com.project.training.MFCCSoundTrainer;
import com.project.training.SoundTrainer;
import com.project.training.SpectrogramSoundTrainer;
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
 * Tests pour la classe BaseSoundTrainer et ses dérivées
 */
public class BaseSoundTrainerTest {
    private static final Logger log = LoggerFactory.getLogger(BaseSoundTrainerTest.class);
    
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
            ensureConfigProperty("sound.num.mfcc", "40");
            ensureConfigProperty("test.num.samples", "10");
            ensureConfigProperty("training.epochs", "1");
            
            // Afficher les valeurs clés pour le débogage
            log.info("Configuration chargée:");
            log.info("sound.input.length = {}", config.getProperty("sound.input.length"));
            log.info("sound.num.mfcc = {}", config.getProperty("sound.num.mfcc"));
            log.info("sound.model.num.classes = {}", config.getProperty("sound.model.num.classes"));
            
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
    public void testSoundTrainerTypes() {
        // Tester les types d'entraîneurs de SoundTrainer
        assertEquals(4, SoundTrainer.SoundTrainerType.values().length);
        assertEquals(SoundTrainer.SoundTrainerType.MFCC, SoundTrainer.SoundTrainerType.valueOf("MFCC"));
        assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM, SoundTrainer.SoundTrainerType.valueOf("SPECTROGRAM"));
        assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16, SoundTrainer.SoundTrainerType.valueOf("SPECTROGRAM_VGG16"));
        assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM_RESNET, SoundTrainer.SoundTrainerType.valueOf("SPECTROGRAM_RESNET"));
    }
    
    @Test
    public void testFactoryMethod() {
        // Tester la création d'un SoundTrainer avec la méthode factory
        SoundTrainer mfccTrainer = SoundTrainer.createTrainer("MFCC", config);
        assertNotNull("Le MFCCSoundTrainer devrait être créé", mfccTrainer);
        assertEquals(SoundTrainer.SoundTrainerType.MFCC, mfccTrainer.getTrainerType());
        
        SoundTrainer spectrogramTrainer = SoundTrainer.createTrainer("SPECTROGRAM", config);
        assertNotNull("Le SpectrogramSoundTrainer devrait être créé", spectrogramTrainer);
        assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM, spectrogramTrainer.getTrainerType());
        
        SoundTrainer vgg16Trainer = SoundTrainer.createTrainer("SPECTROGRAM_VGG16", config);
        assertNotNull("Le VGG16 SpectrogramSoundTrainer devrait être créé", vgg16Trainer);
        assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16, vgg16Trainer.getTrainerType());
    }
    
    @Test
    public void testSetGetTrainerType() {
        // Créer un entraîneur MFCC
        SoundTrainer trainer = new MFCCSoundTrainer(config);
        assertEquals(SoundTrainer.SoundTrainerType.MFCC, trainer.getTrainerType());
        
        // Changer le type
        trainer.setTrainerType(SoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16);
        assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16, trainer.getTrainerType());
    }
    
    @Test
    public void testDataDirectoryStructure() {
        // Test de vérification de la structure des répertoires de données
        String dataDir = config.getProperty("sound.data.dir", "data/raw/sound");
        File dataDirFile = new File(dataDir);
        
        // Si le répertoire n'existe pas, créer une structure minimale pour les tests
        if (!dataDirFile.exists()) {
            log.warn("Le répertoire {} n'existe pas, création d'une structure minimale pour les tests", dataDir);
            
            // Créer le répertoire principal
            assertTrue("Impossible de créer le répertoire de données", dataDirFile.mkdirs());
            
            // Créer quelques sous-répertoires pour les classes d'activités
            String[] activities = {"SLEEPING", "EATING", "CONVERSING", "WATCHING_TV", "LISTENING_MUSIC"};
            for (String activity : activities) {
                File activityDir = new File(dataDirFile, activity);
                assertTrue("Impossible de créer le répertoire d'activité " + activity, 
                        activityDir.mkdir());
            }
        }
        
        // Vérifier que le répertoire existe maintenant
        assertTrue("Le répertoire de données devrait exister", dataDirFile.exists());
        
        // Vérifier qu'il contient des sous-répertoires (classes d'activités)
        File[] activityDirs = dataDirFile.listFiles(File::isDirectory);
        assertNotNull("Le répertoire de données devrait contenir des sous-répertoires", activityDirs);
        assertTrue("Le répertoire de données devrait contenir au moins un sous-répertoire", 
                activityDirs.length > 0);
        
        // Afficher les sous-répertoires trouvés
        log.info("Répertoires d'activités trouvés:");
        for (File dir : activityDirs) {
            log.info("- {}", dir.getName());
        }
    }
}