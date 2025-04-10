package com.project.export;

import com.project.common.utils.DataProcessor;
import com.project.models.activity.ActivityModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Classe pour l'exportation du modèle de détection d'activité.
 * Charge le modèle entraîné et l'exporte dans un format que DL4J peut charger dans d'autres applications.
 */
public class ActivityExporter {
    private static final Logger log = LoggerFactory.getLogger(ActivityExporter.class);
    
    private final ActivityModel model;
    private final String exportPath;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ActivityExporter(Properties config) {
        this.model = new ActivityModel(config);
        this.exportPath = config.getProperty("activity.model.export", "export/activity_model.zip");
        
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
        log.info("Chargement et exportation du modèle de détection d'activité");
        
        try {
            // Charger le modèle entraîné
            model.loadDefaultModel();
            log.info("Modèle chargé avec succès");
            
            // Exporter le modèle
            model.exportModel(exportPath);
            log.info("Modèle exporté avec succès vers {}", exportPath);
            
        } catch (IOException e) {
            log.error("Erreur lors de l'exportation du modèle de détection d'activité", e);
            throw e;
        }
    }
}