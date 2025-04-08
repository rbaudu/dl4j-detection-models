package com.project.common.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Classe utilitaire pour le traitement des données pour tous les modèles.
 * Fournit des méthodes de chargement, normalisation et prétraitement des données.
 */
public class DataProcessor {
    private static final Logger log = LoggerFactory.getLogger(DataProcessor.class);
    
    private DataProcessor() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Charge les données à partir d'un répertoire.
     * @param dataDir Chemin du répertoire de données
     * @return Liste des fichiers de données
     * @throws IOException en cas d'erreur de lecture des fichiers
     */
    public static List<File> loadDataFiles(String dataDir) throws IOException {
        File directory = new File(dataDir);
        if (!directory.exists() || !directory.isDirectory()) {
            throw new IOException("Le répertoire de données n'existe pas : " + dataDir);
        }
        
        List<File> dataFiles = Arrays.asList(directory.listFiles((dir, name) -> 
                name.endsWith(".csv") || name.endsWith(".txt") || name.endsWith(".dat")));
        
        if (dataFiles.isEmpty()) {
            throw new IOException("Aucun fichier de données trouvé dans " + dataDir);
        }
        
        log.info("Chargement de {} fichiers de données depuis {}", dataFiles.size(), dataDir);
        return dataFiles;
    }
    
    /**
     * Crée un dossier s'il n'existe pas déjà.
     * @param dirPath Chemin du dossier à créer
     * @throws IOException en cas d'erreur lors de la création du dossier
     */
    public static void createDirectoryIfNotExists(String dirPath) throws IOException {
        Path path = Paths.get(dirPath);
        if (!Files.exists(path)) {
            Files.createDirectories(path);
            log.info("Répertoire créé : {}", dirPath);
        }
    }
    
    /**
     * Normalise les données entre 0 et 1.
     * @param data Données à normaliser
     * @return Données normalisées
     */
    public static INDArray normalizeData(INDArray data) {
        double min = data.minNumber().doubleValue();
        double max = data.maxNumber().doubleValue();
        
        if (max > min) {
            return data.sub(min).div(max - min);
        } else {
            return data;
        }
    }
    
    /**
     * Divise les données en ensembles d'entraînement et de test.
     * @param dataset Ensemble de données complet
     * @param trainRatio Ratio de données pour l'entraînement (entre 0 et 1)
     * @return Tableau contenant l'ensemble d'entraînement et l'ensemble de test
     */
    public static DataSet[] splitTrainTest(DataSet dataset, double trainRatio) {
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Le ratio d'entraînement doit être entre 0 et 1");
        }
        
        int numExamples = dataset.numExamples();
        int trainSize = (int) Math.round(numExamples * trainRatio);
        int testSize = numExamples - trainSize;
        
        DataSet[] splitData = dataset.splitTestAndTrain(trainSize);
        
        log.info("Données divisées en {} exemples d'entraînement et {} exemples de test", 
                trainSize, testSize);
        
        return splitData;
    }
    
    /**
     * Charge les données CSV dans un INDArray.
     * @param file Fichier CSV à charger
     * @param skipHeader Indique s'il faut ignorer la première ligne (en-tête)
     * @param delimiter Délimiteur utilisé dans le fichier CSV
     * @return INDArray contenant les données chargées
     * @throws IOException en cas d'erreur de lecture du fichier
     */
    public static INDArray loadCsvData(File file, boolean skipHeader, String delimiter) throws IOException {
        List<String> lines = Files.readAllLines(file.toPath());
        
        if (skipHeader && !lines.isEmpty()) {
            lines = lines.subList(1, lines.size());
        }
        
        List<double[]> dataList = new ArrayList<>();
        
        for (String line : lines) {
            String[] values = line.split(delimiter);
            double[] rowData = Arrays.stream(values)
                    .filter(s -> !s.isEmpty())
                    .mapToDouble(Double::parseDouble)
                    .toArray();
            
            if (rowData.length > 0) {
                dataList.add(rowData);
            }
        }
        
        if (dataList.isEmpty()) {
            throw new IOException("Aucune donnée valide trouvée dans le fichier " + file.getName());
        }
        
        int rows = dataList.size();
        int cols = dataList.get(0).length;
        
        INDArray array = Nd4j.create(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            double[] rowData = dataList.get(i);
            for (int j = 0; j < cols; j++) {
                array.putScalar(i, j, rowData[j]);
            }
        }
        
        return array;
    }
    
    /**
     * Vérifie si le chemin spécifié existe et est un répertoire, le crée sinon.
     * @param path Chemin à vérifier/créer
     * @throws IOException en cas d'erreur lors de la création du répertoire
     */
    public static void ensureDirectoryExists(String path) throws IOException {
        File directory = new File(path);
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                throw new IOException("Impossible de créer le répertoire : " + path);
            }
            log.info("Répertoire créé : {}", path);
        } else if (!directory.isDirectory()) {
            throw new IOException("Le chemin existe mais n'est pas un répertoire : " + path);
        }
    }
    
    /**
     * Crée un ensemble de données d'images synthétiques pour l'entraînement.
     * Utile lorsque les vraies données ne sont pas disponibles.
     *
     * @param numExamples Nombre total d'exemples à générer
     * @param height Hauteur des images
     * @param width Largeur des images
     * @param channels Nombre de canaux (3 pour RGB)
     * @param numClasses Nombre de classes
     * @param random Générateur de nombres aléatoires
     * @return DataSet contenant les données synthétiques
     */
    public static DataSet createSyntheticImageDataSet(
            int numExamples, int height, int width, int channels, int numClasses, Random random) {
        
        log.info("Création d'un ensemble de données d'images synthétiques: {}x{}x{}, {} classes, {} exemples",
                height, width, channels, numClasses, numExamples);
        
        // Créer les caractéristiques (images synthétiques)
        INDArray features = Nd4j.rand(new int[]{numExamples, channels, height, width}, random);
        
        // Normaliser entre -1 et 1 pour MobileNetV2
        features = features.mul(2).sub(1);
        
        // Créer les étiquettes one-hot
        INDArray labels = Nd4j.zeros(numExamples, numClasses);
        
        // Attribuer des classes aléatoires, en essayant d'équilibrer les classes
        int examplesPerClass = numExamples / numClasses;
        int remainder = numExamples % numClasses;
        
        int currentIndex = 0;
        for (int classIdx = 0; classIdx < numClasses; classIdx++) {
            int classExamples = examplesPerClass + (classIdx < remainder ? 1 : 0);
            
            for (int i = 0; i < classExamples && currentIndex < numExamples; i++) {
                labels.putScalar(currentIndex, classIdx, 1.0);
                currentIndex++;
            }
        }
        
        // Mélanger les exemples pour éviter tout biais dans l'ordre
        DataSet dataset = new DataSet(features, labels);
        dataset.shuffle();
        
        log.info("Ensemble de données synthétiques créé avec succès");
        
        return dataset;
    }
    
    /**
     * Crée un ensemble de données audio synthétiques pour l'entraînement.
     * Utile lorsque les vraies données ne sont pas disponibles.
     *
     * @param numExamples Nombre total d'exemples à générer
     * @param sampleLength Longueur de l'échantillon audio
     * @param numClasses Nombre de classes
     * @param random Générateur de nombres aléatoires
     * @return DataSet contenant les données synthétiques
     */
    public static DataSet createSyntheticAudioDataSet(
            int numExamples, int sampleLength, int numClasses, Random random) {
        
        log.info("Création d'un ensemble de données audio synthétiques: longueur={}, {} classes, {} exemples",
                sampleLength, numClasses, numExamples);
        
        // Créer les caractéristiques (données audio synthétiques)
        INDArray features = Nd4j.zeros(numExamples, sampleLength);
        
        // Générer des formes d'onde différentes pour chaque classe
        for (int i = 0; i < numExamples; i++) {
            int classIdx = i % numClasses;
            
            // Générer une forme d'onde différente selon la classe
            switch (classIdx) {
                case 0: // Silence (bruit faible)
                    for (int j = 0; j < sampleLength; j++) {
                        features.putScalar(i, j, (random.nextDouble() - 0.5) * 0.1);
                    }
                    break;
                    
                case 1: // Parole (mélange de fréquences)
                    for (int j = 0; j < sampleLength; j++) {
                        double t = j / (double) sampleLength;
                        features.putScalar(i, j, 
                                0.5 * Math.sin(2 * Math.PI * 500 * t) + 
                                0.3 * Math.sin(2 * Math.PI * 1000 * t) +
                                0.2 * (random.nextDouble() - 0.5));
                    }
                    break;
                    
                case 2: // Musique (harmoniques)
                    for (int j = 0; j < sampleLength; j++) {
                        double t = j / (double) sampleLength;
                        features.putScalar(i, j, 
                                0.4 * Math.sin(2 * Math.PI * 440 * t) + 
                                0.3 * Math.sin(2 * Math.PI * 880 * t) +
                                0.2 * Math.sin(2 * Math.PI * 1320 * t) +
                                0.1 * (random.nextDouble() - 0.5));
                    }
                    break;
                    
                case 3: // Bruit ambiant
                    for (int j = 0; j < sampleLength; j++) {
                        features.putScalar(i, j, (random.nextDouble() - 0.5) * 0.6);
                    }
                    break;
                    
                case 4: // Alarme (signal périodique)
                    for (int j = 0; j < sampleLength; j++) {
                        double t = j / (double) sampleLength;
                        double sawtoothWave = 2 * (t * 5 - Math.floor(t * 5 + 0.5));
                        features.putScalar(i, j, sawtoothWave);
                    }
                    break;
                    
                default: // Autres types de sons (générés aléatoirement)
                    double frequency = 200 + (classIdx * 100); // Fréquence différente pour chaque classe
                    for (int j = 0; j < sampleLength; j++) {
                        double t = j / (double) sampleLength;
                        features.putScalar(i, j, 
                                0.7 * Math.sin(2 * Math.PI * frequency * t) +
                                0.3 * (random.nextDouble() - 0.5));
                    }
                    break;
            }
        }
        
        // Normaliser entre -1 et 1
        double maxVal = features.maxNumber().doubleValue();
        double minVal = features.minNumber().doubleValue();
        double maxAbs = Math.max(Math.abs(maxVal), Math.abs(minVal));
        if (maxAbs > 0) {
            features = features.div(maxAbs);
        }
        
        // Créer les étiquettes one-hot
        INDArray labels = Nd4j.zeros(numExamples, numClasses);
        for (int i = 0; i < numExamples; i++) {
            int classIdx = i % numClasses;
            labels.putScalar(i, classIdx, 1.0);
        }
        
        // Mélanger les exemples pour éviter tout biais dans l'ordre
        DataSet dataset = new DataSet(features, labels);
        dataset.shuffle();
        
        log.info("Ensemble de données audio synthétiques créé avec succès");
        
        return dataset;
    }
}
