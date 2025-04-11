package com.project.examples;

import com.project.common.utils.SpectrogramUtils;
import com.project.training.MFCCSoundTrainer;
import com.project.training.SoundTrainer;
import com.project.training.SpectrogramSoundTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Exemple d'utilisation du SoundTrainer refactorisé
 * Montre comment entraîner différents types de modèles et traiter des fichiers audio
 */
public class SoundTrainerExample {
    private static final Logger log = LoggerFactory.getLogger(SoundTrainerExample.class);
    
    /**
     * Point d'entrée principal
     * 
     * @param args Arguments de la ligne de commande
     */
    public static void main(String[] args) {
        try {
            // Chargement de la configuration
            Properties config = loadProperties("config/application.properties");
            
            // Répertoire des données
            String dataDir = config.getProperty("sound.data.dir", "data/raw/sound");
            File dataDirFile = new File(dataDir);
            
            // Vérifier que le répertoire existe
            if (!dataDirFile.exists()) {
                log.error("Le répertoire de données {} n'existe pas", dataDir);
                System.exit(1);
            }
            
            // Choix du modèle à entraîner
            String modelType = args.length > 0 ? args[0] : "mfcc";
            
            switch (modelType.toLowerCase()) {
                case "mfcc":
                    trainMFCCModel(config, dataDir);
                    break;
                    
                case "spectrogram-vgg16":
                    trainSpectrogramVGG16Model(config, dataDir);
                    break;
                    
                case "spectrogram-resnet":
                    trainSpectrogramResNetModel(config, dataDir);
                    break;
                    
                default:
                    log.error("Type de modèle non reconnu: {}. Options valides: mfcc, spectrogram-vgg16, spectrogram-resnet", modelType);
                    System.exit(1);
            }
            
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution de l'exemple", e);
        }
    }
    
    /**
     * Charge un fichier de propriétés
     * 
     * @param path Chemin du fichier
     * @return Propriétés chargées
     */
    private static Properties loadProperties(String path) {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream(path)) {
            props.load(fis);
        } catch (IOException e) {
            log.warn("Impossible de charger le fichier de configuration: {}. Utilisation des valeurs par défaut.", path);
        }
        return props;
    }
    
    /**
     * Entraîne un modèle basé sur MFCC
     * 
     * @param config Configuration
     * @param dataDir Répertoire des données
     */
    private static void trainMFCCModel(Properties config, String dataDir) throws IOException {
        log.info("Entraînement d'un modèle MFCC");
        
        // Configuration pour MFCC
        config.setProperty("sound.model.type", "STANDARD");
        
        // Créer l'entraîneur
        MFCCSoundTrainer trainer = new MFCCSoundTrainer(config);
        
        // Entraîner le modèle
        trainer.trainOnSoundData(dataDir, 0.8);
        
        // Sauvegarder le modèle
        String modelPath = config.getProperty("sound.mfcc.model.path", "models/sound/mfcc_model.zip");
        MultiLayerNetwork model = trainer.getModel();
        model.save(new File(modelPath));
        
        log.info("Modèle MFCC entraîné et sauvegardé dans {}", modelPath);
    }
    
    /**
     * Entraîne un modèle basé sur spectrogrammes avec VGG16
     * 
     * @param config Configuration
     * @param dataDir Répertoire des données
     */
    private static void trainSpectrogramVGG16Model(Properties config, String dataDir) throws IOException {
        log.info("Entraînement d'un modèle Spectrogram VGG16");
        
        // Configuration pour Spectrogram VGG16
        config.setProperty("sound.model.type", "SPECTROGRAM");
        config.setProperty("sound.model.architecture", "VGG16");
        
        // Créer l'entraîneur
        SpectrogramSoundTrainer trainer = new SpectrogramSoundTrainer(config, "VGG16");
        
        // Entraîner le modèle
        trainer.trainOnSoundData(dataDir, 0.8);
        
        // Sauvegarder le modèle
        String modelPath = config.getProperty("sound.spectrogram.model.path", "models/sound/spectrogram_vgg16_model.zip");
        MultiLayerNetwork model = trainer.getModel();
        model.save(new File(modelPath));
        
        log.info("Modèle Spectrogram VGG16 entraîné et sauvegardé dans {}", modelPath);
        
        // Générer quelques exemples de spectrogrammes pour visualisation
        generateExampleSpectrograms(dataDir, config);
    }
    
    /**
     * Entraîne un modèle basé sur spectrogrammes avec ResNet
     * 
     * @param config Configuration
     * @param dataDir Répertoire des données
     */
    private static void trainSpectrogramResNetModel(Properties config, String dataDir) throws IOException {
        log.info("Entraînement d'un modèle Spectrogram ResNet");
        
        // Configuration pour Spectrogram ResNet
        config.setProperty("sound.model.type", "SPECTROGRAM");
        config.setProperty("sound.model.architecture", "ResNet");
        
        // Créer l'entraîneur
        SpectrogramSoundTrainer trainer = new SpectrogramSoundTrainer(config, "ResNet");
        
        // Entraîner le modèle
        trainer.trainOnSoundData(dataDir, 0.8);
        
        // Sauvegarder le modèle
        String modelPath = config.getProperty("sound.spectrogram.model.path", "models/sound/spectrogram_resnet_model.zip");
        MultiLayerNetwork model = trainer.getModel();
        model.save(new File(modelPath));
        
        log.info("Modèle Spectrogram ResNet entraîné et sauvegardé dans {}", modelPath);
        
        // Générer quelques exemples de spectrogrammes pour visualisation
        generateExampleSpectrograms(dataDir, config);
    }
    
    /**
     * Génère des exemples de spectrogrammes pour visualisation
     * 
     * @param dataDir Répertoire des données
     * @param config Configuration
     */
    private static void generateExampleSpectrograms(String dataDir, Properties config) {
        try {
            log.info("Génération d'exemples de spectrogrammes pour visualisation");
            
            // Paramètres des spectrogrammes
            int height = Integer.parseInt(config.getProperty("sound.spectrogram.height", "224"));
            int width = Integer.parseInt(config.getProperty("sound.spectrogram.width", "224"));
            
            // Créer le répertoire de sortie
            String outputDir = "output/spectrograms";
            new File(outputDir).mkdirs();
            
            // Parcourir les fichiers audio
            File dataDirFile = new File(dataDir);
            File[] activityDirs = dataDirFile.listFiles(File::isDirectory);
            
            if (activityDirs != null) {
                // Pour chaque activité, générer un exemple de spectrogramme
                for (File activityDir : activityDirs) {
                    File[] audioFiles = activityDir.listFiles(file -> {
                        String name = file.getName().toLowerCase();
                        return file.isFile() && (name.endsWith(".wav") || name.endsWith(".mp3") || name.endsWith(".ogg"));
                    });
                    
                    if (audioFiles != null && audioFiles.length > 0) {
                        // Prendre le premier fichier comme exemple
                        File audioFile = audioFiles[0];
                        
                        // Générer le spectrogramme
                        BufferedImage spectrogram = SpectrogramUtils.generateSpectrogram(audioFile, height, width);
                        
                        // Sauvegarder le spectrogramme
                        String outputPath = outputDir + "/" + activityDir.getName() + "_example.png";
                        SpectrogramUtils.saveSpectrogramToFile(spectrogram, new File(outputPath));
                        
                        log.info("Spectrogramme généré pour {}: {}", activityDir.getName(), outputPath);
                    }
                }
            }
            
            log.info("Exemples de spectrogrammes générés dans {}", outputDir);
            
        } catch (Exception e) {
            log.error("Erreur lors de la génération des spectrogrammes", e);
        }
    }
}