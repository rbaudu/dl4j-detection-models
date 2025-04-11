package com.project.common.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Utilitaires pour la génération et la manipulation de spectrogrammes audio
 */
public class SpectrogramUtils {
    private static final Logger log = LoggerFactory.getLogger(SpectrogramUtils.class);
    private static final Random random = new Random(42);
    
    /**
     * Génère un spectrogramme à partir d'un fichier audio
     * Dans cette implémentation simplifiée, nous créons un spectrogramme simulé
     * 
     * @param audioFile Fichier audio source
     * @param height Hauteur souhaitée du spectrogramme
     * @param width Largeur souhaitée du spectrogramme
     * @return Image du spectrogramme
     */
    public static BufferedImage generateSpectrogram(File audioFile, int height, int width) {
        log.info("Génération d'un spectrogramme à partir de {} ({} x {})", audioFile.getName(), width, height);
        
        // Dans une implémentation réelle, on utiliserait une bibliothèque comme JavaSound ou JTransforms
        // pour lire l'audio et calculer la FFT pour créer un vrai spectrogramme
        
        // Pour cette version simplifiée, créons une image simulant un spectrogramme
        BufferedImage spectrogram = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = spectrogram.createGraphics();
        
        // Fond noir
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, width, height);
        
        // Simuler des composantes fréquentielles
        // La couleur varie du bleu (fréquences faibles) au rouge (fréquences élevées)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Simuler une valeur d'intensité en fonction de la position et d'une composante aléatoire
                float intensity = (float) Math.sin(x * 0.05) + (float) Math.cos(y * 0.05);
                intensity += random.nextFloat() * 0.3f;
                intensity = Math.abs(intensity);
                intensity = Math.min(1.0f, intensity);
                
                // Créer une couleur en fonction de l'intensité et de la fréquence (y)
                int r = (int) (intensity * 255);
                int g = (int) (intensity * 100);
                int b = (int) (intensity * (255 - y * 255 / height));
                
                Color color = new Color(r, g, b);
                spectrogram.setRGB(x, y, color.getRGB());
            }
        }
        
        // Ajouter quelques lignes horizontales pour simuler des tons purs
        for (int i = 0; i < 5; i++) {
            int y = random.nextInt(height);
            g2d.setColor(new Color(255, 255, 255, 150));
            g2d.drawLine(0, y, width, y);
        }
        
        // Simuler un fondu au début et à la fin
        for (int x = 0; x < width / 10; x++) {
            float alpha = x / (float) (width / 10);
            for (int y = 0; y < height; y++) {
                Color c = new Color(spectrogram.getRGB(x, y), true);
                Color newColor = new Color(c.getRed(), c.getGreen(), c.getBlue(), (int) (alpha * 255));
                spectrogram.setRGB(x, y, newColor.getRGB());
                
                // Aussi pour la fin
                int endX = width - x - 1;
                c = new Color(spectrogram.getRGB(endX, y), true);
                newColor = new Color(c.getRed(), c.getGreen(), c.getBlue(), (int) (alpha * 255));
                spectrogram.setRGB(endX, y, newColor.getRGB());
            }
        }
        
        g2d.dispose();
        return spectrogram;
    }
    
    /**
     * Sauvegarde un spectrogramme dans un fichier image
     * 
     * @param spectrogram Image du spectrogramme
     * @param outputFile Fichier de sortie
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public static void saveSpectrogramToFile(BufferedImage spectrogram, File outputFile) throws IOException {
        // Créer le répertoire parent si nécessaire
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        // Déterminer le format à partir de l'extension du fichier
        String format = "png";
        String fileName = outputFile.getName().toLowerCase();
        if (fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) {
            format = "jpg";
        }
        
        // Sauvegarder l'image
        ImageIO.write(spectrogram, format, outputFile);
        log.info("Spectrogramme sauvegardé dans {}", outputFile.getAbsolutePath());
    }
    
    /**
     * Redimensionne un spectrogramme aux dimensions souhaitées
     * 
     * @param spectrogram Image du spectrogramme
     * @param width Nouvelle largeur
     * @param height Nouvelle hauteur
     * @return Image redimensionnée
     */
    public static BufferedImage resizeSpectrogram(BufferedImage spectrogram, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(spectrogram, 0, 0, width, height, null);
        g2d.dispose();
        return resized;
    }
    
    /**
     * Normalise les valeurs d'intensité d'un spectrogramme pour améliorer le contraste
     * 
     * @param spectrogram Image du spectrogramme
     * @return Image normalisée
     */
    public static BufferedImage normalizeSpectrogram(BufferedImage spectrogram) {
        int width = spectrogram.getWidth();
        int height = spectrogram.getHeight();
        
        // Trouver les valeurs min et max
        int min = 255;
        int max = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color c = new Color(spectrogram.getRGB(x, y));
                int intensity = (c.getRed() + c.getGreen() + c.getBlue()) / 3;
                min = Math.min(min, intensity);
                max = Math.max(max, intensity);
            }
        }
        
        // Éviter la division par zéro
        if (max == min) {
            return spectrogram;
        }
        
        // Normaliser les valeurs
        BufferedImage normalized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color c = new Color(spectrogram.getRGB(x, y));
                int r = normalizeValue(c.getRed(), min, max);
                int g = normalizeValue(c.getGreen(), min, max);
                int b = normalizeValue(c.getBlue(), min, max);
                normalized.setRGB(x, y, new Color(r, g, b).getRGB());
            }
        }
        
        return normalized;
    }
    
    /**
     * Normalise une valeur entre 0 et 255
     */
    private static int normalizeValue(int value, int min, int max) {
        return (int) (255.0 * (value - min) / (max - min));
    }
}