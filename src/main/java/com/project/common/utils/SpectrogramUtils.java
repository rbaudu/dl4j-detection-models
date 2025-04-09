package com.project.common.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Classe utilitaire pour le traitement des spectrogrammes
 * Gère la génération et la manipulation des spectrogrammes pour l'entrée des modèles de ML
 */
public class SpectrogramUtils {
    private static final Logger log = LoggerFactory.getLogger(SpectrogramUtils.class);
    
    // Paramètres par défaut pour les spectrogrammes
    private static final int DEFAULT_HEIGHT = 224;
    private static final int DEFAULT_WIDTH = 224;
    private static final int DEFAULT_CHANNELS = 1;
    
    /**
     * Génère un spectrogramme à partir d'un fichier audio
     * 
     * @param audioFile Fichier audio
     * @param height Hauteur du spectrogramme
     * @param width Largeur du spectrogramme
     * @return Image du spectrogramme
     */
    public static BufferedImage generateSpectrogram(File audioFile, int height, int width) {
        return AudioUtils.generateSpectrogram(audioFile, height, width);
    }
    
    /**
     * Sauvegarde une image de spectrogramme sur le disque
     * 
     * @param spectrogram Image du spectrogramme
     * @param outputFile Fichier de sortie
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public static void saveSpectrogramToFile(BufferedImage spectrogram, File outputFile) throws IOException {
        // Créer les répertoires parents si nécessaire
        if (!outputFile.getParentFile().exists()) {
            outputFile.getParentFile().mkdirs();
        }
        
        // Sauvegarder l'image au format PNG
        ImageIO.write(spectrogram, "PNG", outputFile);
        log.debug("Spectrogramme sauvegardé dans {}", outputFile.getAbsolutePath());
    }
    
    /**
     * Convertit un spectrogramme en INDArray pour l'entrée du modèle
     * 
     * @param spectrogram Image du spectrogramme
     * @param height Hauteur cible
     * @param width Largeur cible
     * @param channels Nombre de canaux (généralement 1 pour niveaux de gris)
     * @return INDArray au format [1, channels, height, width]
     */
    public static INDArray spectrogramToINDArray(BufferedImage spectrogram, int height, int width, int channels) {
        // Redimensionner l'image si nécessaire
        BufferedImage resized = spectrogram;
        if (spectrogram.getWidth() != width || spectrogram.getHeight() != height) {
            resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            resized.getGraphics().drawImage(spectrogram.getScaledInstance(width, height, java.awt.Image.SCALE_SMOOTH), 0, 0, null);
        }
        
        // Créer un tableau pour contenir les données
        float[] data = new float[channels * height * width];
        
        // Pour une image en niveaux de gris (1 canal)
        if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = resized.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    
                    // Convertir RGB en niveau de gris et normaliser entre 0 et 1
                    float grayValue = (r * 0.299f + g * 0.587f + b * 0.114f) / 255.0f;
                    data[y * width + x] = grayValue;
                }
            }
            
            // Créer l'INDArray avec la forme [1, channels, height, width]
            return Nd4j.create(data).reshape(1, channels, height, width);
        } 
        // Pour une image couleur (3 canaux)
        else if (channels == 3) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = resized.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    
                    // Normaliser entre 0 et 1 et stocker dans le tableau
                    data[(0 * height * width) + (y * width + x)] = r / 255.0f;
                    data[(1 * height * width) + (y * width + x)] = g / 255.0f;
                    data[(2 * height * width) + (y * width + x)] = b / 255.0f;
                }
            }
            
            // Créer l'INDArray avec la forme [1, channels, height, width]
            return Nd4j.create(data).reshape(1, channels, height, width);
        } else {
            throw new IllegalArgumentException("Nombre de canaux non supporté: " + channels);
        }
    }
    
    /**
     * Crée un spectrogramme à partir d'un fichier audio et le convertit directement en INDArray
     * 
     * @param audioFile Fichier audio
     * @param height Hauteur du spectrogramme
     * @param width Largeur du spectrogramme
     * @param channels Nombre de canaux
     * @return INDArray pour l'entrée du modèle
     */
    public static INDArray audioToSpectrogramINDArray(File audioFile, int height, int width, int channels) {
        // Générer le spectrogramme
        BufferedImage spectrogram = generateSpectrogram(audioFile, height, width);
        
        // Convertir en INDArray
        return spectrogramToINDArray(spectrogram, height, width, channels);
    }
    
    /**
     * Traite tous les fichiers audio d'un répertoire en spectrogrammes et les sauvegarde
     * 
     * @param sourceDir Répertoire source contenant les fichiers audio
     * @param outputDir Répertoire de sortie pour les spectrogrammes
     * @param height Hauteur des spectrogrammes
     * @param width Largeur des spectrogrammes
     * @return Nombre de spectrogrammes générés
     */
    public static int processDirectoryToSpectrograms(File sourceDir, File outputDir, int height, int width) {
        if (!sourceDir.exists() || !sourceDir.isDirectory()) {
            log.error("Le répertoire source '{}' n'existe pas ou n'est pas un répertoire", sourceDir.getAbsolutePath());
            return 0;
        }
        
        // Créer le répertoire de sortie s'il n'existe pas
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        int count = 0;
        
        // Traiter les fichiers audio
        File[] audioFiles = sourceDir.listFiles(file -> {
            String name = file.getName().toLowerCase();
            return file.isFile() && (name.endsWith(".wav") || name.endsWith(".mp3") || name.endsWith(".ogg"));
        });
        
        if (audioFiles != null) {
            for (File audioFile : audioFiles) {
                try {
                    // Générer le spectrogramme
                    BufferedImage spectrogram = generateSpectrogram(audioFile, height, width);
                    
                    // Créer le fichier de sortie avec le même nom mais extension .png
                    String outputName = audioFile.getName();
                    outputName = outputName.substring(0, outputName.lastIndexOf('.')) + ".png";
                    File outputFile = new File(outputDir, outputName);
                    
                    // Sauvegarder le spectrogramme
                    saveSpectrogramToFile(spectrogram, outputFile);
                    
                    count++;
                } catch (Exception e) {
                    log.error("Erreur lors du traitement du fichier {}", audioFile.getName(), e);
                }
            }
        }
        
        // Traiter les sous-répertoires récursivement
        File[] subDirs = sourceDir.listFiles(File::isDirectory);
        if (subDirs != null) {
            for (File subDir : subDirs) {
                File subOutputDir = new File(outputDir, subDir.getName());
                count += processDirectoryToSpectrograms(subDir, subOutputDir, height, width);
            }
        }
        
        return count;
    }
    
    /**
     * Extrait les valeurs RVB moyennes d'un spectrogramme par régions
     * Utile pour créer des caractéristiques simplifiées à partir d'un spectrogramme
     * 
     * @param spectrogram Image du spectrogramme
     * @param numRegionsX Nombre de régions horizontales
     * @param numRegionsY Nombre de régions verticales
     * @return Tableau de caractéristiques
     */
    public static float[] extractRegionFeatures(BufferedImage spectrogram, int numRegionsX, int numRegionsY) {
        int width = spectrogram.getWidth();
        int height = spectrogram.getHeight();
        
        int regionWidth = width / numRegionsX;
        int regionHeight = height / numRegionsY;
        
        float[] features = new float[numRegionsX * numRegionsY * 3];  // 3 pour R, G, B
        
        for (int ry = 0; ry < numRegionsY; ry++) {
            for (int rx = 0; rx < numRegionsX; rx++) {
                int startX = rx * regionWidth;
                int startY = ry * regionHeight;
                int endX = Math.min(startX + regionWidth, width);
                int endY = Math.min(startY + regionHeight, height);
                
                // Calculer les moyennes RVB pour cette région
                float sumR = 0, sumG = 0, sumB = 0;
                int pixelCount = 0;
                
                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        int rgb = spectrogram.getRGB(x, y);
                        sumR += (rgb >> 16) & 0xFF;
                        sumG += (rgb >> 8) & 0xFF;
                        sumB += rgb & 0xFF;
                        pixelCount++;
                    }
                }
                
                // Calculer les moyennes et normaliser
                int featureIndex = (ry * numRegionsX + rx) * 3;
                features[featureIndex] = sumR / (pixelCount * 255.0f);
                features[featureIndex + 1] = sumG / (pixelCount * 255.0f);
                features[featureIndex + 2] = sumB / (pixelCount * 255.0f);
            }
        }
        
        return features;
    }
    
    /**
     * Combine les images de spectrogrammes d'une classe en une moyenne visuelle
     * Utile pour visualiser les modèles caractéristiques d'une classe
     * 
     * @param spectrogramFiles Tableau des fichiers de spectrogramme
     * @param height Hauteur du spectrogramme de sortie
     * @param width Largeur du spectrogramme de sortie
     * @return Image moyennée des spectrogrammes
     */
    public static BufferedImage createAverageSpectrogram(File[] spectrogramFiles, int height, int width) {
        if (spectrogramFiles == null || spectrogramFiles.length == 0) {
            return new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        }
        
        // Tableaux pour stocker les sommes des valeurs RGB
        long[][] sumR = new long[height][width];
        long[][] sumG = new long[height][width];
        long[][] sumB = new long[height][width];
        
        int validFileCount = 0;
        
        // Accumuler les valeurs RGB
        for (File file : spectrogramFiles) {
            try {
                BufferedImage img = ImageIO.read(file);
                if (img == null) continue;
                
                // Redimensionner si nécessaire
                BufferedImage resized = img;
                if (img.getWidth() != width || img.getHeight() != height) {
                    resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    resized.getGraphics().drawImage(img.getScaledInstance(width, height, java.awt.Image.SCALE_SMOOTH), 0, 0, null);
                }
                
                // Accumuler les valeurs RGB
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int rgb = resized.getRGB(x, y);
                        sumR[y][x] += (rgb >> 16) & 0xFF;
                        sumG[y][x] += (rgb >> 8) & 0xFF;
                        sumB[y][x] += rgb & 0xFF;
                    }
                }
                
                validFileCount++;
            } catch (IOException e) {
                log.error("Erreur lors de la lecture du fichier {}", file.getName(), e);
            }
        }
        
        // Créer l'image moyennée
        BufferedImage avgImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        if (validFileCount > 0) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int avgR = (int) (sumR[y][x] / validFileCount);
                    int avgG = (int) (sumG[y][x] / validFileCount);
                    int avgB = (int) (sumB[y][x] / validFileCount);
                    
                    int avgRGB = (avgR << 16) | (avgG << 8) | avgB;
                    avgImg.setRGB(x, y, avgRGB);
                }
            }
        }
        
        return avgImg;
    }
}
