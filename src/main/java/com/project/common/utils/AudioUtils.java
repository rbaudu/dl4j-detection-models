package com.project.common.utils;

import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AudioUtils {
    
    /**
     * Charge un fichier audio et le convertit en format approprié
     * @param filePath Chemin du fichier audio
     * @return AudioInputStream traité
     */
    public static AudioInputStream loadAudioFile(String filePath) throws IOException, UnsupportedAudioFileException {
        File audioFile = new File(filePath);
        AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(audioFile);
        
        // Conversion vers un format standard si nécessaire
        AudioFormat format = audioInputStream.getFormat();
        if (format.getEncoding() != AudioFormat.Encoding.PCM_SIGNED) {
            AudioFormat targetFormat = new AudioFormat(
                    AudioFormat.Encoding.PCM_SIGNED,
                    format.getSampleRate(),
                    16,
                    format.getChannels(),
                    format.getChannels() * 2,
                    format.getSampleRate(),
                    false);
            
            // Utilisation de la méthode correcte pour convertir le format audio
            audioInputStream = AudioSystem.getAudioInputStream(targetFormat, audioInputStream);
        }
        
        return audioInputStream;
    }
    
    /**
     * Convertit un audio en tableau de double[]
     * @param audioInputStream Le flux audio à convertir
     * @return Les données audio sous forme de tableau de doubles
     */
    public static double[] audioInputStreamToDoubleArray(AudioInputStream audioInputStream) throws IOException {
        AudioFormat format = audioInputStream.getFormat();
        long frameLength = audioInputStream.getFrameLength();
        int frameSize = format.getFrameSize();
        int channels = format.getChannels();
        
        byte[] audioBytes = new byte[(int) (frameLength * frameSize)];
        int bytesRead = audioInputStream.read(audioBytes);
        
        double[] audioData = new double[bytesRead / (frameSize / channels)];
        
        int sampleIndex = 0;
        for (int i = 0; i < bytesRead; i += frameSize / channels) {
            double sampleValue = 0;
            
            if (format.getSampleSizeInBits() == 16) {
                sampleValue = ((audioBytes[i] & 0xFF) | (audioBytes[i + 1] << 8)) / 32768.0;
            } else if (format.getSampleSizeInBits() == 8) {
                sampleValue = (audioBytes[i] & 0xFF) / 128.0 - 1.0;
            }
            
            audioData[sampleIndex++] = sampleValue;
        }
        
        return audioData;
    }
    
    /**
     * Convertit les données audio en INDArray pour traitement DL4J
     * @param audioData Données audio
     * @return INDArray adapté pour DL4J
     */
    public static INDArray audioToFeatures(double[] audioData, int windowSize, int hopSize) {
        int numWindows = (audioData.length - windowSize) / hopSize + 1;
        
        // Créer un tableau pour stocker les caractéristiques
        INDArray features = Nd4j.zeros(numWindows, windowSize);
        
        for (int i = 0; i < numWindows; i++) {
            int startIndex = i * hopSize;
            
            // Copier la fenêtre actuelle dans l'INDArray
            double[] window = new double[windowSize];
            System.arraycopy(audioData, startIndex, window, 0, windowSize);
            
            // Appliquer une fenêtre de Hamming
            applyHammingWindow(window);
            
            // Stocker les caractéristiques
            for (int j = 0; j < windowSize; j++) {
                features.putScalar(i, j, window[j]);
            }
        }
        
        return features;
    }
    
    /**
     * Applique une fenêtre de Hamming aux données audio
     * @param window Les données audio à traiter
     */
    private static void applyHammingWindow(double[] window) {
        for (int i = 0; i < window.length; i++) {
            double multiplier = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (window.length - 1));
            window[i] = window[i] * multiplier;
        }
    }
    
    /**
     * Extrait les caractéristiques MFCC d'un fichier audio
     * Cette méthode remplace l'utilisation de WaveFileLoader qui n'est pas disponible
     */
    public static INDArray extractMFCCFeatures(String filePath, int numCoefficients) throws IOException, UnsupportedAudioFileException {
        // Charger le fichier audio
        AudioInputStream audioInputStream = loadAudioFile(filePath);
        
        // Convertir en tableau de doubles
        double[] audioData = audioInputStreamToDoubleArray(audioInputStream);
        
        // Paramètres pour l'extraction MFCC
        int windowSize = 512;
        int hopSize = 256;
        
        // Créer une représentation fenêtrée du signal
        INDArray frames = audioToFeatures(audioData, windowSize, hopSize);
        
        // Simuler l'extraction de MFCC
        // Note: Dans un cas réel, vous devriez utiliser une bibliothèque dédiée pour l'extraction MFCC
        // comme TarsosDSP ou implémenter l'algorithme complet
        int numFrames = frames.rows();
        INDArray mfccs = Nd4j.zeros(numFrames, numCoefficients);
        
        // Pour cet exemple, nous remplissons simplement avec des valeurs simulées
        for (int i = 0; i < numFrames; i++) {
            INDArray frame = frames.getRow(i);
            
            // Simuler l'extraction MFCC - à remplacer par une véritable implémentation
            for (int j = 0; j < numCoefficients; j++) {
                mfccs.putScalar(i, j, frame.sumNumber().doubleValue() / (j + 1));
            }
        }
        
        return mfccs;
    }
}