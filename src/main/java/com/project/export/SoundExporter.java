package com.project.export;

import com.project.common.utils.DataProcessor;
import com.project.models.sound.SoundModel;
import com.project.models.sound.SpectrogramSoundModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Classe pour l'exportation du modèle de détection de sons.
 * Charge le modèle entraîné et l'exporte dans un format que DL4J peut charger dans d'autres applications.
 */
public class SoundExporter {
    private static final Logger log = LoggerFactory.getLogger(SoundExporter.class);
    
    private final Properties config;
    private final String exportPath;
    private final String soundModelType;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public SoundExporter(Properties config) {
        this.config = config;
        this.soundModelType = config.getProperty("sound.model.type", "STANDARD");
        
        if ("SPECTROGRAM".equalsIgnoreCase(soundModelType)) {
            this.exportPath = config.getProperty("sound.spectrogram.model.export", "export/spectrogram_sound_model.zip");
        } else {
            this.exportPath = config.getProperty("sound.model.export", "export/sound_model.zip");
        }
        
        try {
            // S'assurer que le répertoire d'export existe
            String exportDir = exportPath.substring(0, exportPath.lastIndexOf('/'));
            DataProcessor.ensureDirectoryExists(exportDir);
        } catch (IOException e) {
            log.error("Erreur lors de la création du répertoire d'export", e);
        }
    }
    
    /**
     * Exporte le modèle au format DL4J.
     *
     * @throws IOException en cas d'erreur lors de l'exportation
     */
    public void export() throws IOException {
        if ("SPECTROGRAM".equalsIgnoreCase(soundModelType)) {
            exportSpectrogramModel();
        } else {
            exportStandardModel();
        }
    }
    
    /**
     * Exporte le modèle standard de détection de sons au format DL4J.
     *
     * @throws IOException en cas d'erreur lors de l'exportation
     */
    private void exportStandardModel() throws IOException {
        log.info("Chargement et exportation du modèle standard de détection de sons");
        
        try {
            // Charger le modèle entraîné
            SoundModel model = new SoundModel(config);
            model.loadDefaultModel();
            log.info("Modèle standard chargé avec succès");
            
            // Exporter le modèle
            model.exportModel(exportPath);
            log.info("Modèle standard exporté avec succès vers {}", exportPath);
            
        } catch (IOException e) {
            log.error("Erreur lors de l'exportation du modèle standard de détection de sons", e);
            throw e;
        }
    }
    
    /**
     * Exporte le modèle de détection de sons basé sur spectrogrammes au format DL4J.
     *
     * @throws IOException en cas d'erreur lors de l'exportation
     */
    private void exportSpectrogramModel() throws IOException {
        log.info("Chargement et exportation du modèle de détection de sons basé sur spectrogrammes");
        
        try {
            // Charger le modèle entraîné
            SpectrogramSoundModel model = new SpectrogramSoundModel(config);
            model.loadDefaultModel();
            log.info("Modèle basé sur spectrogrammes chargé avec succès");
            
            // Exporter le modèle
            model.exportModel(exportPath);
            log.info("Modèle basé sur spectrogrammes exporté avec succès vers {}", exportPath);
            
        } catch (IOException e) {
            log.error("Erreur lors de l'exportation du modèle de détection de sons basé sur spectrogrammes", e);
            throw e;
        }
    }
}
