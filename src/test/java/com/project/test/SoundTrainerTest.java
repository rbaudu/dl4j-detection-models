package com.project.test;

import com.project.training.MFCCSoundTrainer;
import com.project.training.SoundTrainer;
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
 * Tests d'intégration pour la nouvelle architecture SoundTrainer
 */
public class SoundTrainerTest {
    private static final Logger log = LoggerFactory.getLogger(SoundTrainerTest.class);
    
    private Properties config;
    
    @Before
    public void setUp() {
        // Charger la configuration
        config = new Properties();
        try {
            config.load(new FileInputStream("config/application.properties"));
            
            // Configurer pour les tests
            config.setProperty("sound.model.num.classes", "5");  // Réduire pour les tests
            config.setProperty("test.num.samples", "10");  // Réduire pour les tests
            config.setProperty("training.epochs", "1");  // Juste un seul epoch pour les tests
            
        } catch (IOException e) {
            log.warn("Configuration par défaut utilisée", e);
        }
    }
    
    @Test
    public void testSoundTrainerFacade() {
        // Tester la création d'un SoundTrainer (façade)
        SoundTrainer trainer = new SoundTrainer(config);
        assertNotNull("Le SoundTrainer devrait être créé", trainer);
        
        // Vérifier que le type par défaut est correctement déterminé
        assertEquals("Le type devrait être déterminé à partir de la configuration", 
                config.getProperty("sound.model.type", "STANDARD").equals("SPECTROGRAM") ? 
                        SoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16 : 
                        SoundTrainer.SoundTrainerType.MFCC, 
                trainer.getTrainerType());
        
        // Tester le changement de type
        SoundTrainer.SoundTrainerType newType = trainer.getTrainerType() == SoundTrainer.SoundTrainerType.MFCC ?
                SoundTrainer.SoundTrainerType.SPECTROGRAM_VGG16 : SoundTrainer.SoundTrainerType.MFCC;
        
        trainer.setTrainerType(newType);
        assertEquals("Le type devrait être changé", newType, trainer.getTrainerType());
    }
    
    @Test
    public void testMFCCSoundTrainer() {
        // Tester la création d'un MFCCSoundTrainer
        MFCCSoundTrainer trainer = new MFCCSoundTrainer(config);
        assertNotNull("Le MFCCSoundTrainer devrait être créé", trainer);
        
        // Initialiser le modèle
        trainer.initializeModel();
        
        // Vérifier que le modèle est créé
        MultiLayerNetwork model = trainer.getModel();
        assertNotNull("Le modèle devrait être créé", model);
        
        // Vérifier les propriétés du modèle
        int inputSize = Integer.parseInt(config.getProperty("sound.input.length", "16000")) * 
                        Integer.parseInt(config.getProperty("sound.num.mfcc", "40"));
        
        int[] inputShape = model.getLayerInputShape(0);
        assertEquals("La taille d'entrée du modèle devrait correspondre aux paramètres", 
                inputSize, inputShape[1]);
    }
    
    @Test
    public void testSpectrogramSoundTrainer() {
        // Configurer pour l'approche spectrogramme
        config.setProperty("sound.model.type", "SPECTROGRAM");
        config.setProperty("sound.model.architecture", "VGG16");
        
        // Tester la création d'un SpectrogramSoundTrainer
        SpectrogramSoundTrainer trainer = new SpectrogramSoundTrainer(config);
        assertNotNull("Le SpectrogramSoundTrainer devrait être créé", trainer);
        
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
