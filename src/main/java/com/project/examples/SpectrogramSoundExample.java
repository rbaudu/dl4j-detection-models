package com.project.examples;

import com.project.models.sound.SpectrogramSoundModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

/**
 * Exemple d'utilisation du modèle de sons basé sur spectrogrammes.
 * Cette classe démontre comment entraîner et utiliser le modèle pour
 * classifier des sons à partir de fichiers audio.
 */
public class SpectrogramSoundExample {
    private static final Logger log = LoggerFactory.getLogger(SpectrogramSoundExample.class);
    
    public static void main(String[] args) {
        try {
            // Charger la configuration
            Properties config = loadConfiguration("config/application.properties");
            
            // Créer le modèle
            SpectrogramSoundModel soundModel = new SpectrogramSoundModel(config);
            
            // Chemin vers le dataset d'entraînement
            String datasetPath = config.getProperty("sound.train.data.dir", "data/raw/sound");
            File datasetDir = new File(datasetPath);
            
            if (!datasetDir.exists()) {
                log.warn("Le répertoire de données {} n'existe pas. Création de la structure de répertoires...", datasetPath);
                createSampleDatasetStructure(datasetPath);
            }
            
            // Entraîner ou charger le modèle
            String modelPath = config.getProperty("sound.spectrogram.model.path", "models/sound/spectrogram_model.zip");
            File modelFile = new File(modelPath);
            
            if (modelFile.exists() && !getBooleanProperty(config, "sound.force.retrain", false)) {
                // Charger le modèle existant
                log.info("Chargement du modèle existant depuis {}", modelPath);
                soundModel.loadModel(modelPath);
            } else {
                // Initialiser un nouveau modèle
                log.info("Initialisation d'un nouveau modèle");
                soundModel.initNewModel();
                
                // Entraîner le modèle
                int epochs = Integer.parseInt(config.getProperty("sound.model.epochs", "100"));
                int batchSize = Integer.parseInt(config.getProperty("sound.model.batch.size", "32"));
                
                log.info("Entraînement du modèle sur {} avec {} époques et une taille de batch de {}", 
                        datasetPath, epochs, batchSize);
                soundModel.trainOnDataset(datasetPath, epochs, batchSize);
            }
            
            // Exemple d'utilisation du modèle pour la prédiction
            demonstratePrediction(soundModel, datasetPath);
            
            // Création de spectrogrammes pour visualisation
            demonstrateSpectrogramGeneration(soundModel, datasetPath);
            
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution des exemples", e);
        }
    }
    
    /**
     * Démontre l'utilisation du modèle pour prédire la classe de fichiers audio.
     * 
     * @param soundModel Modèle de sons
     * @param datasetPath Chemin vers le dataset
     * @throws IOException Si une erreur survient lors de la prédiction
     */
    private static void demonstratePrediction(SpectrogramSoundModel soundModel, String datasetPath) throws IOException {
        log.info("=== Démonstration de la prédiction sur des fichiers audio ===");
        
        // Trouver quelques fichiers audio pour la prédiction
        File datasetDir = new File(datasetPath);
        File[] classDirs = datasetDir.listFiles(File::isDirectory);
        
        if (classDirs == null || classDirs.length == 0) {
            log.warn("Aucun sous-répertoire de classe trouvé, impossible de faire des prédictions");
            return;
        }
        
        // Choisir un fichier de chaque classe pour la prédiction
        for (File classDir : classDirs) {
            File[] audioFiles = classDir.listFiles(file -> 
                    file.isFile() && (file.getName().endsWith(".wav") || 
                                     file.getName().endsWith(".mp3") ||
                                     file.getName().endsWith(".ogg")));
            
            if (audioFiles != null && audioFiles.length > 0) {
                // Prendre le premier fichier pour la démonstration
                File audioFile = audioFiles[0];
                
                // Faire la prédiction
                String predictedClass = soundModel.predict(audioFile.getPath());
                
                // Afficher le résultat
                log.info("Fichier: {} (classe réelle: {}) -> Classe prédite: {}", 
                        audioFile.getName(), classDir.getName(), predictedClass);
            }
        }
    }
    
    /**
     * Démontre la génération de spectrogrammes à partir de fichiers audio.
     * 
     * @param soundModel Modèle de sons
     * @param datasetPath Chemin vers le dataset
     * @throws IOException Si une erreur survient lors de la génération
     */
    private static void demonstrateSpectrogramGeneration(SpectrogramSoundModel soundModel, String datasetPath) throws IOException {
        log.info("=== Démonstration de la génération de spectrogrammes ===");
        
        // Créer un répertoire pour les spectrogrammes
        Path spectrogramsDir = Paths.get("output/spectrograms");
        Files.createDirectories(spectrogramsDir);
        
        // Trouver quelques fichiers audio pour la génération
        File datasetDir = new File(datasetPath);
        File[] classDirs = datasetDir.listFiles(File::isDirectory);
        
        if (classDirs == null || classDirs.length == 0) {
            log.warn("Aucun sous-répertoire de classe trouvé, impossible de générer des spectrogrammes");
            return;
        }
        
        // Choisir un fichier de chaque classe pour la génération
        for (File classDir : classDirs) {
            File[] audioFiles = classDir.listFiles(file -> 
                    file.isFile() && (file.getName().endsWith(".wav") || 
                                     file.getName().endsWith(".mp3") ||
                                     file.getName().endsWith(".ogg")));
            
            if (audioFiles != null && audioFiles.length > 0) {
                // Prendre le premier fichier pour la démonstration
                File audioFile = audioFiles[0];
                
                // Générer le spectrogramme
                java.awt.image.BufferedImage spectrogram = soundModel.generateSpectrogram(audioFile.getPath());
                
                // Sauvegarder le spectrogramme
                String outputPath = spectrogramsDir.resolve(
                        classDir.getName() + "_" + audioFile.getName().replace(".", "_") + ".png").toString();
                
                javax.imageio.ImageIO.write(spectrogram, "PNG", new File(outputPath));
                
                log.info("Spectrogramme généré et sauvegardé dans {}", outputPath);
            }
        }
        
        log.info("Les spectrogrammes ont été générés et sauvegardés dans {}", spectrogramsDir);
    }
    
    /**
     * Crée une structure de répertoires exemple pour les données audio.
     * 
     * @param datasetPath Chemin vers le dataset
     * @throws IOException Si une erreur survient lors de la création
     */
    private static void createSampleDatasetStructure(String datasetPath) throws IOException {
        log.info("Création d'une structure de répertoires exemple pour les données audio");
        
        // Définir les classes d'activité
        String[] activityClasses = {
            "CONVERSING",
            "EATING",
            "COOKING",
            "CLEANING",
            "MOVING",
            "UNKNOWN"
        };
        
        // Créer les répertoires pour chaque classe
        for (String activityClass : activityClasses) {
            Path classDir = Paths.get(datasetPath, activityClass);
            Files.createDirectories(classDir);
            
            log.info("Répertoire créé: {}", classDir);
        }
        
        log.info("Structure de répertoires créée avec succès. Veuillez y ajouter des fichiers audio (.wav, .mp3, .ogg)");
    }
    
    /**
     * Charge le fichier de configuration.
     * 
     * @param configPath Chemin vers le fichier de configuration
     * @return Propriétés de configuration
     * @throws IOException Si une erreur survient lors du chargement
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
     * Récupère une propriété booléenne à partir de la configuration.
     * 
     * @param config Propriétés de configuration
     * @param key Clé de la propriété
     * @param defaultValue Valeur par défaut
     * @return Valeur de la propriété
     */
    private static boolean getBooleanProperty(Properties config, String key, boolean defaultValue) {
        String value = config.getProperty(key);
        return (value != null) ? Boolean.parseBoolean(value) : defaultValue;
    }
}
