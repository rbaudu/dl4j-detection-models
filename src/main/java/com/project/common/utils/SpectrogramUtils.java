package com.project.common.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
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
        try {
            // Charger le fichier audio
            javax.sound.sampled.AudioInputStream audioStream = AudioUtils.loadWavFile(audioFile);
            
            // Convertir en mono si nécessaire
            audioStream = AudioUtils.convertToMono(audioStream);
            
            // Obtenir les données audio
            byte[] audioData = readAllBytes(audioStream);
            javax.sound.sampled.AudioFormat format = audioStream.getFormat();
            
            // Convertir en valeurs à virgule flottante
            float[] samples = bytesToFloats(audioData, format);
            
            // Génération du spectrogramme
            return createSpectrogramImage(samples, format.getSampleRate(), height, width);
        } catch (Exception e) {
            log.error("Erreur lors de la génération du spectrogramme pour " + audioFile.getName(), e);
            // En cas d'erreur, retourner une image noire
            return new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        }
    }
    
    /**
     * Lit tous les octets d'un flux audio
     * 
     * @param audioStream Flux audio à lire
     * @return Tableau d'octets contenant les données audio
     * @throws IOException Si une erreur survient lors de la lecture
     */
    private static byte[] readAllBytes(javax.sound.sampled.AudioInputStream audioStream) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[16384];
        while ((nRead = audioStream.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        buffer.flush();
        return buffer.toByteArray();
    }
    
    /**
     * Convertit des octets audio en valeurs à virgule flottante
     * 
     * @param audioData Données audio en octets
     * @param format Format audio
     * @return Tableau de valeurs à virgule flottante
     */
    private static float[] bytesToFloats(byte[] audioData, javax.sound.sampled.AudioFormat format) {
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int numSamples = audioData.length / bytesPerSample;
        float[] samples = new float[numSamples];
        
        // Audio 16 bits
        if (format.getSampleSizeInBits() == 16) {
            java.nio.ShortBuffer shortBuffer = java.nio.ByteBuffer.wrap(audioData)
                .order(format.isBigEndian() ? java.nio.ByteOrder.BIG_ENDIAN : java.nio.ByteOrder.LITTLE_ENDIAN)
                .asShortBuffer();
            
            for (int i = 0; i < numSamples; i++) {
                samples[i] = shortBuffer.get(i) / 32768.0f;  // Normaliser entre -1 et 1
            }
        } 
        // Audio 8 bits non signé
        else if (format.getSampleSizeInBits() == 8) {
            for (int i = 0; i < numSamples; i++) {
                samples[i] = ((audioData[i] & 0xff) - 128) / 128.0f;  // Normaliser entre -1 et 1
            }
        }
        
        return samples;
    }
    
    /**
     * Crée une image de spectrogramme à partir d'échantillons audio
     * 
     * @param samples Échantillons audio
     * @param sampleRate Taux d'échantillonnage
     * @param height Hauteur de l'image
     * @param width Largeur de l'image
     * @return Image du spectrogramme
     */
    private static BufferedImage createSpectrogramImage(float[] samples, float sampleRate, int height, int width) {
        // Simuler un spectrogramme simple
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        // Si l'échantillon est vide, retourner une image noire
        if (samples.length == 0) {
            return image;
        }
        
        // Calculer une représentation simplifiée du spectrogramme
        // En réalité, vous utiliseriez FFT et d'autres techniques
        for (int x = 0; x < width; x++) {
            int startIdx = (int) (x * samples.length / (float) width);
            int endIdx = (int) ((x + 1) * samples.length / (float) width);
            
            // Pour chaque point temporel, calculer l'énergie pour différentes bandes de fréquence
            for (int y = 0; y < height; y++) {
                // Simuler différentes bandes de fréquence
                int band = height - 1 - y;  // Inverser pour avoir les basses fréquences en bas
                
                // Calculer l'énergie pour cette bande de fréquence (simulée)
                float energy = 0;
                for (int i = startIdx; i < endIdx && i < samples.length; i++) {
                    // Plus la bande est haute, plus la fréquence est élevée
                    energy += Math.abs(samples[i] * Math.sin(Math.PI * band / height));
                }
                
                // Normaliser et convertir en couleur
                energy = Math.min(1.0f, energy * 5);  // Amplifier et limiter
                int colorValue = (int) (energy * 255);
                
                // Créer une couleur basée sur l'énergie
                int color = getColorFromValue(energy);
                
                image.setRGB(x, y, color);
            }
        }
        
        return image;
    }
    
    /**
     * Convertit une valeur normalisée (0-1) en une couleur
     * Utilise une colormap de type "viridis"
     * 
     * @param value Valeur normalisée entre 0 et 1
     * @return Couleur RGB
     */
    private static int getColorFromValue(float value) {
        // Colormap simplifiée de type "viridis"
        // De bleu foncé (faible) à jaune vif (élevé)
        float r, g, b;
        
        if (value < 0.25) {
            r = 0;
            g = 4 * value;
            b = 1;
        } else if (value < 0.5) {
            r = 0;
            g = 1;
            b = 1 - 4 * (value - 0.25f);
        } else if (value < 0.75) {
            r = 4 * (value - 0.5f);
            g = 1;
            b = 0;
        } else {
            r = 1;
            g = 1 - 4 * (value - 0.75f);
            b = 0;
        }
        
        int ri = (int) (r * 255);
        int gi = (int) (g * 255);
        int bi = (int) (b * 255);
        
        return (ri << 16) | (gi << 8) | bi;
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
}
