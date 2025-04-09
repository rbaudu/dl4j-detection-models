package com.project.common.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.sound.sampled.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Classe utilitaire pour le traitement des fichiers audio
 * Gère l'extraction de caractéristiques MFCC et la génération de spectrogrammes
 */
public class AudioUtils {
    private static final Logger log = LoggerFactory.getLogger(AudioUtils.class);
    
    // Paramètres par défaut pour le traitement audio
    private static final float DEFAULT_SAMPLE_RATE = 44100f;
    private static final int DEFAULT_FFT_SIZE = 2048;
    private static final int DEFAULT_HOP_SIZE = 512;
    private static final int DEFAULT_MEL_BANDS = 128;
    
    /**
     * Charge un fichier WAV et retourne le flux audio
     * 
     * @param file Fichier WAV à charger
     * @return Flux audio
     * @throws IOException Si une erreur survient lors de la lecture du fichier
     * @throws UnsupportedAudioFileException Si le format du fichier n'est pas supporté
     */
    public static AudioInputStream loadWavFile(File file) throws IOException, UnsupportedAudioFileException {
        AudioInputStream audioStream = AudioSystem.getAudioInputStream(file);
        
        // Convertir en format PCM signé 16 bits si nécessaire
        AudioFormat format = audioStream.getFormat();
        if (format.getEncoding() != AudioFormat.Encoding.PCM_SIGNED || format.getSampleSizeInBits() != 16) {
            AudioFormat targetFormat = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                format.getSampleRate(),
                16,
                format.getChannels(),
                format.getChannels() * 2,
                format.getSampleRate(),
                false);
            audioStream = AudioSystem.getAudioInputStream(targetFormat, audioStream);
        }
        
        return audioStream;
    }
    
    /**
     * Convertit un flux audio stéréo en mono si nécessaire
     * 
     * @param audioStream Flux audio à convertir
     * @return Flux audio mono
     * @throws IOException Si une erreur survient lors de la conversion
     */
    public static AudioInputStream convertToMono(AudioInputStream audioStream) throws IOException {
        AudioFormat format = audioStream.getFormat();
        
        // Si déjà mono, retourner le flux tel quel
        if (format.getChannels() == 1) {
            return audioStream;
        }
        
        // Lire les données audio
        byte[] data = readAllBytes(audioStream);
        
        // Créer une nouvelle format audio mono
        AudioFormat monoFormat = new AudioFormat(
            format.getSampleRate(),
            format.getSampleSizeInBits(),
            1,
            format.getSampleSizeInBits() == 8 ? false : true,
            false);
        
        // Convertir les données en mono
        byte[] monoData = stereoToMono(data, format);
        
        // Créer un nouveau flux audio mono
        ByteArrayInputStream bais = new ByteArrayInputStream(monoData);
        return new AudioInputStream(bais, monoFormat, monoData.length / monoFormat.getFrameSize());
    }
    
    /**
     * Convertit des données audio stéréo en mono
     * 
     * @param stereoData Données stéréo
     * @param format Format audio des données
     * @return Données audio mono
     */
    private static byte[] stereoToMono(byte[] stereoData, AudioFormat format) {
        int channels = format.getChannels();
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int bytesPerFrame = bytesPerSample * channels;
        int monoLength = stereoData.length / channels;
        byte[] monoData = new byte[monoLength];
        
        // Pour l'audio 16 bits
        if (bytesPerSample == 2) {
            ShortBuffer stereoBuffer = ByteBuffer.wrap(stereoData)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asShortBuffer();
            ShortBuffer monoBuffer = ByteBuffer.wrap(monoData)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asShortBuffer();
            
            int numFrames = stereoData.length / bytesPerFrame;
            for (int i = 0; i < numFrames; i++) {
                int sum = 0;
                for (int j = 0; j < channels; j++) {
                    sum += stereoBuffer.get(i * channels + j);
                }
                monoBuffer.put(i, (short) (sum / channels));
            }
        } 
        // Pour l'audio 8 bits
        else if (bytesPerSample == 1) {
            int numFrames = stereoData.length / bytesPerFrame;
            for (int i = 0; i < numFrames; i++) {
                int sum = 0;
                for (int j = 0; j < channels; j++) {
                    sum += stereoData[i * bytesPerFrame + j];
                }
                monoData[i] = (byte) (sum / channels);
            }
        }
        
        return monoData;
    }
    
    /**
     * Lit tous les octets d'un flux audio
     * 
     * @param audioStream Flux audio à lire
     * @return Tableau d'octets contenant les données audio
     * @throws IOException Si une erreur survient lors de la lecture
     */
    private static byte[] readAllBytes(AudioInputStream audioStream) throws IOException {
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
     * Extrait les coefficients MFCC d'un fichier audio
     * 
     * @param file Fichier audio
     * @param numMfcc Nombre de coefficients MFCC à extraire
     * @param inputLength Longueur maximale de la séquence d'entrée
     * @return Matrice de coefficients MFCC [1, inputLength * numMfcc]
     */
    public static INDArray extractMFCC(File file, int numMfcc, int inputLength) {
        try {
            // Charger le fichier WAV
            AudioInputStream audioStream = loadWavFile(file);
            
            // Convertir en mono si nécessaire
            audioStream = convertToMono(audioStream);
            
            // Obtenir les données audio
            byte[] audioData = readAllBytes(audioStream);
            AudioFormat format = audioStream.getFormat();
            
            // Convertir en valeurs à virgule flottante
            float[] samples = bytesToFloats(audioData, format);
            
            // Extraction des MFCC (simulée)
            // Note: Dans une implémentation réelle, utilisez une bibliothèque comme librosa ou JLibrosa
            float[][] mfccFeatures = simulateMFCCExtraction(samples, numMfcc, inputLength);
            
            // Convertir en INDArray
            INDArray features = Nd4j.create(flattenMFCC(mfccFeatures, numMfcc, inputLength));
            return features.reshape(1, inputLength * numMfcc);
            
        } catch (Exception e) {
            log.error("Erreur lors de l'extraction des MFCC pour " + file.getName(), e);
            // En cas d'erreur, retourner un vecteur de zéros
            return Nd4j.zeros(1, inputLength * numMfcc);
        }
    }
    
    /**
     * Simule l'extraction des MFCC
     * Dans une implémentation réelle, utilisez une bibliothèque comme librosa ou JLibrosa
     * 
     * @param samples Échantillons audio
     * @param numMfcc Nombre de coefficients MFCC
     * @param inputLength Longueur de la séquence
     * @return Matrice des coefficients MFCC
     */
    private static float[][] simulateMFCCExtraction(float[] samples, int numMfcc, int inputLength) {
        // Simulation simple de l'extraction de MFCC
        // En réalité, vous utiliseriez une bibliothèque comme librosa ou JLibrosa
        
        float[][] mfcc = new float[inputLength][numMfcc];
        
        // Diviser l'audio en segments et extraire les MFCC pour chaque segment
        int samplesPerSegment = samples.length / inputLength;
        for (int i = 0; i < Math.min(inputLength, samples.length / samplesPerSegment); i++) {
            int startIdx = i * samplesPerSegment;
            
            // Calculer l'énergie du segment
            double energy = 0;
            for (int j = 0; j < samplesPerSegment && startIdx + j < samples.length; j++) {
                energy += samples[startIdx + j] * samples[startIdx + j];
            }
            energy = Math.sqrt(energy / samplesPerSegment);
            
            // Remplir les coefficients MFCC avec des valeurs dérivées de l'énergie
            // C'est une simulation très simple - dans la réalité, ces valeurs seraient calculées
            // à partir de la transformée de Fourier et d'autres transformations
            for (int j = 0; j < numMfcc; j++) {
                mfcc[i][j] = (float) (energy * Math.sin(j * Math.PI / numMfcc));
            }
        }
        
        return mfcc;
    }
    
    /**
     * Aplatit une matrice MFCC en un vecteur
     * 
     * @param mfcc Matrice MFCC [inputLength][numMfcc]
     * @param numMfcc Nombre de coefficients MFCC
     * @param inputLength Longueur de la séquence
     * @return Vecteur MFCC aplati
     */
    private static float[] flattenMFCC(float[][] mfcc, int numMfcc, int inputLength) {
        float[] flatMfcc = new float[inputLength * numMfcc];
        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < numMfcc; j++) {
                flatMfcc[i * numMfcc + j] = i < mfcc.length ? mfcc[i][j] : 0;
            }
        }
        return flatMfcc;
    }
    
    /**
     * Convertit des octets audio en valeurs à virgule flottante
     * 
     * @param audioData Données audio en octets
     * @param format Format audio
     * @return Tableau de valeurs à virgule flottante
     */
    private static float[] bytesToFloats(byte[] audioData, AudioFormat format) {
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int numSamples = audioData.length / bytesPerSample;
        float[] samples = new float[numSamples];
        
        // Audio 16 bits
        if (format.getSampleSizeInBits() == 16) {
            ShortBuffer shortBuffer = ByteBuffer.wrap(audioData)
                .order(format.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN)
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
     * Crée une carte des classes à partir d'un répertoire contenant des sous-dossiers par activité
     * 
     * @param rootDir Répertoire racine contenant les sous-dossiers d'activités
     * @return Map associant des indices aux noms de classes
     */
    public static Map<Integer, String> createLabelMap(File rootDir) {
        Map<Integer, String> labelMap = new HashMap<>();
        
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            log.error("Le répertoire racine '{}' n'existe pas ou n'est pas un répertoire", rootDir.getAbsolutePath());
            return labelMap;
        }
        
        File[] activityDirs = rootDir.listFiles(File::isDirectory);
        if (activityDirs == null) {
            log.error("Impossible de lister les sous-répertoires dans '{}'", rootDir.getAbsolutePath());
            return labelMap;
        }
        
        // Trier les répertoires par nom pour une cohérence des indices
        Arrays.sort(activityDirs);
        
        // Créer la carte des labels
        for (int i = 0; i < activityDirs.length; i++) {
            labelMap.put(i, activityDirs[i].getName());
            log.info("Classe {}: {}", i, activityDirs[i].getName());
        }
        
        return labelMap;
    }
}
