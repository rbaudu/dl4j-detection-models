package com.project.common.config;

import org.junit.Before;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Tests unitaires pour la classe ConfigValidator
 */
public class ConfigValidatorTest {
    
    private Properties validConfig;
    private Properties invalidConfig;
    
    @Before
    public void setUp() {
        // Préparer une configuration valide
        validConfig = new Properties();
        validConfig.setProperty("data.root.dir", "data");
        validConfig.setProperty("models.root.dir", "models");
        validConfig.setProperty("presence.model.type", "YOLO");
        validConfig.setProperty("presence.model.num.classes", "2");
        validConfig.setProperty("activity.model.type", "VGG16");
        validConfig.setProperty("activity.model.num.classes", "27");
        validConfig.setProperty("activity.model.input.height", "224");
        validConfig.setProperty("activity.model.input.width", "224");
        validConfig.setProperty("sound.model.type", "SPECTROGRAM");
        validConfig.setProperty("sound.model.num.classes", "8");
        validConfig.setProperty("sound.spectrogram.height", "224");
        validConfig.setProperty("sound.spectrogram.width", "224");
        validConfig.setProperty("sound.sample.rate", "44100");
        validConfig.setProperty("sound.fft.size", "2048");
        
        // Préparer une configuration invalide
        invalidConfig = new Properties();
        // Manquant des propriétés requises
    }
    
    @Test
    public void testValidConfiguration() {
        assertTrue("La configuration valide devrait passer la validation", 
                  ConfigValidator.validateConfig(validConfig));
    }
    
    @Test
    public void testInvalidConfiguration() {
        assertFalse("La configuration invalide ne devrait pas passer la validation", 
                   ConfigValidator.validateConfig(invalidConfig));
    }
    
    @Test
    public void testInvalidNumClasses() {
        // Configurer un nombre de classes invalide pour le modèle d'activité
        Properties config = new Properties();
        // Copier toutes les propriétés valides
        for (String key : validConfig.stringPropertyNames()) {
            config.setProperty(key, validConfig.getProperty(key));
        }
        config.setProperty("activity.model.num.classes", "0");
        
        assertFalse("La configuration avec un nombre de classes invalide ne devrait pas passer", 
                   ConfigValidator.validateConfig(config));
    }
    
    @Test
    public void testInvalidModelType() {
        // Configurer un type de modèle inconnu (mais devrait être traité comme avertissement, pas erreur)
        Properties config = new Properties();
        // Copier toutes les propriétés valides
        for (String key : validConfig.stringPropertyNames()) {
            config.setProperty(key, validConfig.getProperty(key));
        }
        config.setProperty("activity.model.type", "UNKNOWN_TYPE");
        
        assertTrue("La configuration avec un type de modèle inconnu devrait passer avec avertissement", 
                 ConfigValidator.validateConfig(config));
    }
    
    @Test
    public void testInvalidImageDimensions() {
        // Configurer des dimensions d'image négatives
        Properties config = new Properties();
        // Copier toutes les propriétés valides
        for (String key : validConfig.stringPropertyNames()) {
            config.setProperty(key, validConfig.getProperty(key));
        }
        config.setProperty("activity.model.input.height", "-1");
        
        assertFalse("La configuration avec des dimensions d'image négatives ne devrait pas passer", 
                   ConfigValidator.validateConfig(config));
    }
    
    @Test
    public void testMissingRequiredPath() {
        // Supprimer une propriété de chemin requise
        Properties config = new Properties();
        // Copier toutes les propriétés valides sauf data.root.dir
        for (String key : validConfig.stringPropertyNames()) {
            if (!key.equals("data.root.dir")) {
                config.setProperty(key, validConfig.getProperty(key));
            }
        }
        
        assertFalse("La configuration avec un chemin requis manquant ne devrait pas passer", 
                   ConfigValidator.validateConfig(config));
    }
}