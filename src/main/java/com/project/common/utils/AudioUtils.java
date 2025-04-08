package com.project.common.utils;

import org.datavec.audio.loader.WaveFileLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Classe utilitaire pour le traitement des fichiers audio.
 * Fournit des méthodes pour charger et prétraiter les fichiers audio pour différents modèles.
 */
public class AudioUtils {
    private static final Logger log = LoggerFactory.getLogger(AudioUtils.class);
    
    // Paramètres standards pour YAMNet
    public static final int YAMNET_SAMPLE_RATE = 16000;
    public static final int YAMNET_WAVEFORM_LENGTH = 16000; // 1 seconde d'audio à 16kHz
    
    private AudioUtils() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Charge et prétraite un fichier audio pour le modèle YAMNet.
     * - Rééchantillonne l'audio à 16kHz si nécessaire
     * - Convertit en mono si nécessaire
     * - Normalise l'amplitude entre -1 et 1
     *
     * @param audioFile Fichier audio à charger
     * @return INDArray contenant les données audio prétraitées
     * @throws IOException en cas d'erreur lors du chargement du fichier audio
     */
    public static INDArray preprocessAudioForYAMNet(File audioFile) throws IOException {
        log.debug("Prétraitement du fichier audio {} pour YAMNet", audioFile.getName());
        
        try {
            // Charger le fichier audio
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(audioFile);
            AudioFormat format = audioInputStream.getFormat();
            
            log.debug("Format audio d'origine: taux d'échantillonnage={}, canaux={}, bits par échantillon={}",
                    format.getSampleRate(), format.getChannels(), format.getSampleSizeInBits());
            
            // Convertir en format compatible avec YAMNet si nécessaire (mono, 16kHz)
            if (format.getSampleRate() != YAMNET_SAMPLE_RATE || format.getChannels() > 1) {
                AudioFormat targetFormat = new AudioFormat(
                        YAMNET_SAMPLE_RATE, 16, 1, true, false);
                audioInputStream = AudioSystem.getAudioFormat(targetFormat, audioInputStream);
                format = audioInputStream.getFormat();
                log.debug("Format audio converti: taux d'échantillonnage={}, canaux={}, bits par échantillon={}",
                        format.getSampleRate(), format.getChannels(), format.getSampleSizeInBits());
            }
            
            // Lire les données audio
            byte[] audioBytes = new byte[audioInputStream.available()];
            int bytesRead = audioInputStream.read(audioBytes);
            log.debug("Lecture de {} octets de données audio", bytesRead);
            
            // Convertir les octets en valeurs d'échantillons
            float[] audioSamples = bytesToFloatArray(audioBytes, format);
            
            // Limiter à YAMNET_WAVEFORM_LENGTH échantillons ou ajouter du silence si nécessaire
            float[] processedAudio = new float[YAMNET_WAVEFORM_LENGTH];
            if (audioSamples.length >= YAMNET_WAVEFORM_LENGTH) {
                // Prendre seulement le début si l'audio est trop long
                System.arraycopy(audioSamples, 0, processedAudio, 0, YAMNET_WAVEFORM_LENGTH);
            } else {
                // Copier l'audio disponible et remplir le reste avec des zéros
                System.arraycopy(audioSamples, 0, processedAudio, 0, audioSamples.length);
                Arrays.fill(processedAudio, audioSamples.length, YAMNET_WAVEFORM_LENGTH, 0.0f);
            }
            
            // Convertir en INDArray
            INDArray audioArray = Nd4j.create(processedAudio);
            
            return audioArray;
            
        } catch (UnsupportedAudioFileException e) {
            log.error("Format audio non pris en charge pour le fichier {}", audioFile.getName(), e);
            throw new IOException("Format audio non pris en charge", e);
        }
    }
    
    /**
     * Convertit un tableau d'octets audio en tableau de valeurs flottantes entre -1 et 1.
     *
     * @param audioBytes Tableau d'octets audio
     * @param format Format audio
     * @return Tableau de valeurs flottantes normalisées
     */
    private static float[] bytesToFloatArray(byte[] audioBytes, AudioFormat format) {
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int sampleCount = audioBytes.length / bytesPerSample;
        float[] samples = new float[sampleCount];
        
        float maxValue;
        switch (format.getSampleSizeInBits()) {
            case 8:
                maxValue = 128.0f;
                break;
            case 16:
                maxValue = 32768.0f;
                break;
            case 24:
                maxValue = 8388608.0f;
                break;
            case 32:
                maxValue = 2147483648.0f;
                break;
            default:
                maxValue = 32768.0f;
        }
        
        for (int i = 0; i < sampleCount; i++) {
            int sampleValue = 0;
            
            // Combiner les octets en une valeur d'échantillon
            if (format.isBigEndian()) {
                for (int b = 0; b < bytesPerSample; b++) {
                    sampleValue = (sampleValue << 8) | (audioBytes[i * bytesPerSample + b] & 0xFF);
                }
            } else {
                for (int b = bytesPerSample - 1; b >= 0; b--) {
                    sampleValue = (sampleValue << 8) | (audioBytes[i * bytesPerSample + b] & 0xFF);
                }
            }
            
            // Normaliser entre -1 et 1
            samples[i] = sampleValue / maxValue;
        }
        
        return samples;
    }
    
    /**
     * Vérifie si un fichier est un fichier audio valide basé sur son extension.
     *
     * @param file Fichier à vérifier
     * @return true si le fichier est un fichier audio, false sinon
     */
    public static boolean isAudioFile(File file) {
        if (file == null || !file.isFile()) {
            return false;
        }
        
        String name = file.getName().toLowerCase();
        return name.endsWith(".wav") || name.endsWith(".mp3") || 
               name.endsWith(".ogg") || name.endsWith(".flac") || 
               name.endsWith(".aac") || name.endsWith(".m4a");
    }
    
    /**
     * Charge un fichier audio WAV en utilisant WaveFileLoader de DataVec.
     *
     * @param waveFile Fichier WAV à charger
     * @return Tableau de données audio
     * @throws IOException en cas d'erreur lors du chargement du fichier
     */
    public static double[] loadWaveFile(File waveFile) throws IOException {
        WaveFileLoader loader = new WaveFileLoader();
        return loader.loadWaveAsDouble(waveFile);
    }
}
