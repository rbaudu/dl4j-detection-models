package com.project.common.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

/**
 * Classe utilitaire pour charger et préparer les données audio
 * Gère le chargement des fichiers et la création des ensembles d'entraînement
 */
public class DataLoaderUtils {
    private static final Logger log = LoggerFactory.getLogger(DataLoaderUtils.class);
    
    // Extensions de fichiers audio supportées
    private static final String[] SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg"};
    
    /**
     * Charge les fichiers audio à partir d'un répertoire et de ses sous-répertoires
     * 
     * @param rootDir Répertoire racine contenant les sous-répertoires d'activités
     * @param extensions Extensions de fichiers à charger (par défaut: wav, mp3, ogg)
     * @return Liste des fichiers audio trouvés
     */
    public static List<File> loadAudioFiles(String rootDir, String[] extensions) {
        List<File> audioFiles = new ArrayList<>();
        String[] supportedExtensions = (extensions != null && extensions.length > 0) ?
                extensions : SUPPORTED_AUDIO_EXTENSIONS;
        
        File root = new File(rootDir);
        if (!root.exists() || !root.isDirectory()) {
            log.error("Le répertoire racine '{}' n'existe pas ou n'est pas un répertoire", rootDir);
            return audioFiles;
        }
        
        // Parcourir les sous-répertoires (classes d'activités)
        File[] activityDirs = root.listFiles(File::isDirectory);
        if (activityDirs == null) {
            log.error("Impossible de lister les sous-répertoires dans '{}'", rootDir);
            return audioFiles;
        }
        
        // Pour chaque répertoire d'activité, trouver les fichiers audio
        for (File activityDir : activityDirs) {
            File[] files = activityDir.listFiles(file -> {
                if (!file.isFile()) return false;
                String name = file.getName().toLowerCase();
                for (String ext : supportedExtensions) {
                    if (name.endsWith(ext)) return true;
                }
                return false;
            });
            
            if (files != null) {
                audioFiles.addAll(Arrays.asList(files));
                log.info("Chargement de {} fichiers audio depuis {}", files.length, activityDir.getName());
            }
        }
        
        return audioFiles;
    }
    
    /**
     * Divise un ensemble de données en ensembles d'entraînement et de test
     * 
     * @param dataSet Ensemble de données complet
     * @param trainRatio Ratio pour l'ensemble d'entraînement (0-1)
     * @return Tableau avec [entraînement, test]
     */
    public static DataSet[] createTrainTestSplit(DataSet dataSet, double trainRatio) {
        int numExamples = dataSet.numExamples();
        int trainSize = (int) (numExamples * trainRatio);
        int testSize = numExamples - trainSize;
        
        // Mélanger les données
        dataSet.shuffle();
        
        DataSet trainData = new DataSet();
        DataSet testData = new DataSet();
        
        // Extraire les sous-ensembles
        for (int i = 0; i < numExamples; i++) {
            DataSet example = dataSet.get(i);
            if (i < trainSize) {
                if (trainData.isEmpty()) {
                    trainData = example.copy();
                } else {
                    trainData = DataSet.merge(Arrays.asList(trainData, example));
                }
            } else {
                if (testData.isEmpty()) {
                    testData = example.copy();
                } else {
                    testData = DataSet.merge(Arrays.asList(testData, example));
                }
            }
        }
        
        log.info("Division des données: {} exemples d'entraînement, {} exemples de test", trainSize, testSize);
        
        return new DataSet[] { trainData, testData };
    }
    
    /**
     * Extrait les étiquettes des répertoires d'activités
     * 
     * @param rootDir Répertoire racine contenant les sous-répertoires d'activités
     * @return Map associant les noms de classes à des indices
     */
    public static Map<String, Integer> extractLabelsFromDirectories(String rootDir) {
        Map<String, Integer> labelMap = new HashMap<>();
        
        File root = new File(rootDir);
        if (!root.exists() || !root.isDirectory()) {
            log.error("Le répertoire racine '{}' n'existe pas ou n'est pas un répertoire", rootDir);
            return labelMap;
        }
        
        // Parcourir les sous-répertoires (classes d'activités)
        File[] activityDirs = root.listFiles(File::isDirectory);
        if (activityDirs == null) {
            log.error("Impossible de lister les sous-répertoires dans '{}'", rootDir);
            return labelMap;
        }
        
        // Trier les répertoires par nom pour une cohérence des indices
        Arrays.sort(activityDirs, Comparator.comparing(File::getName));
        
        // Créer la carte des labels
        for (int i = 0; i < activityDirs.length; i++) {
            String className = activityDirs[i].getName();
            labelMap.put(className, i);
            log.info("Classe {}: {} (indice {})", i, className, i);
        }
        
        return labelMap;
    }
    
    /**
     * Détermine la classe d'activité à partir du chemin du fichier
     * 
     * @param filePath Chemin du fichier
     * @return Nom de la classe d'activité
     */
    public static String getActivityClassFromPath(String filePath) {
        File file = new File(filePath);
        File parent = file.getParentFile();
        
        if (parent != null) {
            return parent.getName();
        }
        
        return "UNKNOWN";
    }
    
    /**
     * Crée un DataSet à partir de fichiers audio avec des caractéristiques MFCC
     * 
     * @param audioFiles Liste des fichiers audio
     * @param labelMap Map associant les noms de classes à des indices
     * @param numMfcc Nombre de coefficients MFCC
     * @param inputLength Longueur de la séquence d'entrée
     * @return DataSet contenant les caractéristiques et les étiquettes
     */
    public static DataSet createMFCCDataSet(List<File> audioFiles, Map<String, Integer> labelMap, int numMfcc, int inputLength) {
        int numClasses = labelMap.size();
        
        List<INDArray> featuresList = new ArrayList<>();
        List<INDArray> labelsList = new ArrayList<>();
        
        for (File file : audioFiles) {
            try {
                // Extraire les caractéristiques MFCC
                INDArray features = AudioUtils.extractMFCC(file, numMfcc, inputLength);
                
                // Déterminer la classe à partir du chemin du fichier
                String className = getActivityClassFromPath(file.getAbsolutePath());
                Integer classIndex = labelMap.get(className);
                
                if (classIndex == null) {
                    log.warn("Classe inconnue pour le fichier {}: {}", file.getName(), className);
                    continue;
                }
                
                // Créer l'étiquette (one-hot encoding)
                INDArray label = Nd4j.zeros(1, numClasses);
                label.putScalar(0, classIndex, 1.0);
                
                // Ajouter aux listes
                featuresList.add(features);
                labelsList.add(label);
                
            } catch (Exception e) {
                log.error("Erreur lors du traitement du fichier {}", file.getName(), e);
            }
        }
        
        // Créer le DataSet complet
        if (featuresList.isEmpty() || labelsList.isEmpty()) {
            log.warn("Aucune donnée valide pour créer le DataSet");
            return new DataSet();
        }
        
        INDArray featuresArray = Nd4j.vstack(featuresList);
        INDArray labelsArray = Nd4j.vstack(labelsList);
        
        log.info("DataSet créé avec {} exemples", featuresList.size());
        
        return new DataSet(featuresArray, labelsArray);
    }
    
    /**
     * Crée un DataSet à partir de fichiers audio avec des spectrogrammes
     * 
     * @param audioFiles Liste des fichiers audio
     * @param labelMap Map associant les noms de classes à des indices
     * @param height Hauteur du spectrogramme
     * @param width Largeur du spectrogramme
     * @param channels Nombre de canaux (généralement 1 pour les spectrogrammes)
     * @return DataSet contenant les caractéristiques et les étiquettes
     */
    public static DataSet createSpectrogramDataSet(List<File> audioFiles, Map<String, Integer> labelMap, int height, int width, int channels) {
        int numClasses = labelMap.size();
        
        List<INDArray> featuresList = new ArrayList<>();
        List<INDArray> labelsList = new ArrayList<>();
        
        for (File file : audioFiles) {
            try {
                // Générer le spectrogramme et le convertir en INDArray
                INDArray features = SpectrogramUtils.audioToSpectrogramINDArray(file, height, width, channels);
                
                // Déterminer la classe à partir du chemin du fichier
                String className = getActivityClassFromPath(file.getAbsolutePath());
                Integer classIndex = labelMap.get(className);
                
                if (classIndex == null) {
                    log.warn("Classe inconnue pour le fichier {}: {}", file.getName(), className);
                    continue;
                }
                
                // Créer l'étiquette (one-hot encoding)
                INDArray label = Nd4j.zeros(1, numClasses);
                label.putScalar(0, classIndex, 1.0);
                
                // Ajouter aux listes
                featuresList.add(features);
                labelsList.add(label);
                
                log.debug("Fichier traité: {}, classe: {}", file.getName(), className);
                
            } catch (Exception e) {
                log.error("Erreur lors du traitement du fichier {}", file.getName(), e);
            }
        }
        
        // Créer le DataSet complet
        if (featuresList.isEmpty() || labelsList.isEmpty()) {
            log.warn("Aucune donnée valide pour créer le DataSet");
            return new DataSet();
        }
        
        INDArray featuresArray = Nd4j.vstack(featuresList);
        INDArray labelsArray = Nd4j.vstack(labelsList);
        
        log.info("DataSet créé avec {} exemples", featuresList.size());
        
        return new DataSet(featuresArray, labelsArray);
    }
}
