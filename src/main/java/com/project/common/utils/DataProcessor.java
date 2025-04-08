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
}
