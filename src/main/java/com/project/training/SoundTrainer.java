package com.project.training;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import javax.sound.sampled.UnsupportedAudioFileException;

import com.project.common.utils.AudioUtils;

public class SoundTrainer extends ModelTrainer {

    private int featureSize;
    private int numClasses;
    private Random rng;

    public SoundTrainer(int batchSize, int numEpochs, String modelOutputPath, int featureSize, int numClasses) {
        super(batchSize, numEpochs, modelOutputPath);
        this.featureSize = featureSize;
        this.numClasses = numClasses;
        this.rng = new Random(42);
    }

    /**
     * Prépare et charge les données audio pour l'entraînement
     * @param audioDir Répertoire contenant les fichiers audio
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareAudioData(String audioDir, double trainTestRatio) throws IOException, UnsupportedAudioFileException {
        // Charger tous les fichiers audio du répertoire
        File audioDirFile = new File(audioDir);
        File[] audioFiles = audioDirFile.listFiles((dir, name) -> 
            name.toLowerCase().endsWith(".wav") || 
            name.toLowerCase().endsWith(".mp3") || 
            name.toLowerCase().endsWith(".aiff")
        );

        if (audioFiles == null || audioFiles.length == 0) {
            throw new IOException("Aucun fichier audio trouvé dans le répertoire: " + audioDir);
        }

        // Lire et traiter chaque fichier audio
        List<INDArray> featuresList = new ArrayList<>();
        List<INDArray> labelsList = new ArrayList<>();

        for (File audioFile : audioFiles) {
            // Extraire la classe à partir du nom du fichier ou de la structure du répertoire
            int classLabel = extractClassLabel(audioFile);
            
            // Extraire les caractéristiques MFCC
            INDArray features = AudioUtils.extractMFCCFeatures(audioFile.getAbsolutePath(), featureSize);
            
            // Créer un vecteur one-hot pour la classe
            INDArray label = Nd4j.zeros(1, numClasses);
            label.putScalar(0, classLabel, 1.0);
            
            // Ajouter aux listes
            featuresList.add(features);
            labelsList.add(label);
        }

        // Fusionner tous les exemples en un seul DataSet
        INDArray allFeatures = Nd4j.vstack(featuresList);
        INDArray allLabels = Nd4j.vstack(labelsList);
        DataSet allData = new DataSet(allFeatures, allLabels);
        allData.shuffle();

        // Diviser en ensembles d'entraînement et de test
        int trainSize = (int) (allData.numExamples() * trainTestRatio);
        int testSize = allData.numExamples() - trainSize;

        DataSet trainData = allData.get(0, trainSize);
        DataSet testData = allData.get(trainSize, testSize);

        return new DataSet[] { trainData, testData };
    }

    /**
     * Prépare et charge les données de caractéristiques audio à partir de fichiers CSV
     * @param featuresFile Fichier CSV contenant les caractéristiques extraites
     * @param labelsFile Fichier CSV contenant les étiquettes
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareFeatureData(String featuresFile, String labelsFile, double trainTestRatio) throws IOException, InterruptedException {
        // Configurer les lecteurs de fichiers CSV pour les caractéristiques et les étiquettes
        RecordReader featuresReader = new CSVRecordReader();
        featuresReader.initialize(new FileSplit(new File(featuresFile)));

        RecordReader labelsReader = new CSVRecordReader();
        labelsReader.initialize(new FileSplit(new File(labelsFile)));

        // Créer un itérateur pour les données
        DataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("features", featuresReader)
                .addReader("labels", labelsReader)
                .addInput("features")
                .addOutputOneHot("labels", 0, numClasses)
                .build();

        // Collecter toutes les données
        List<DataSet> dataList = new ArrayList<>();
        while (iterator.hasNext()) {
            dataList.add(iterator.next());
        }

        // Fusionner tous les mini-lots en un seul DataSet
        DataSet allData = DataSet.merge(dataList);
        allData.shuffle();

        // Diviser en ensembles d'entraînement et de test
        int trainSize = (int) (allData.numExamples() * trainTestRatio);
        int testSize = allData.numExamples() - trainSize;

        DataSet trainData = allData.get(0, trainSize);
        DataSet testData = allData.get(trainSize, testSize);

        return new DataSet[] { trainData, testData };
    }

    /**
     * Initialise et entraîne le modèle pour la classification de son
     * @param audioDir Répertoire contenant les fichiers audio
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnAudioData(String audioDir, double trainTestRatio) throws IOException, UnsupportedAudioFileException {
        // Préparer les données
        DataSet[] data = prepareAudioData(audioDir, trainTestRatio);
        DataSet trainData = data[0];
        DataSet testData = data[1];

        // Initialiser le modèle
        initializeModel(featureSize, numClasses);

        // Entraîner le modèle
        train(trainData, testData);
    }

    /**
     * Extrait l'étiquette de classe à partir du fichier audio
     * (Cette méthode devrait être adaptée selon votre structure de données)
     * @param audioFile Le fichier audio
     * @return L'index de la classe
     */
    private int extractClassLabel(File audioFile) {
        // Cette implémentation est un exemple et devrait être adaptée à votre cas d'utilisation
        String filename = audioFile.getName();
        
        // Exemple: si le nom du fichier est "class_1_sample.wav", nous extrayons 1 comme classe
        // Cette logique peut être modifiée selon votre convention de nommage
        String[] parts = filename.split("_");
        if (parts.length >= 2) {
            try {
                return Integer.parseInt(parts[1]);
            } catch (NumberFormatException e) {
                // En cas d'erreur, retourner 0 comme classe par défaut
                return 0;
            }
        }
        
        return 0;  // Classe par défaut
    }
}

/**
 * Classe utilitaire pour créer un DataSetIterator à partir de plusieurs RecordReaders
 */
class RecordReaderMultiDataSetIterator {
    
    public static class Builder {
        private int batchSize;
        private final List<RecordReader> readers = new ArrayList<>();
        private final List<String> readerNames = new ArrayList<>();
        private final List<String> inputNames = new ArrayList<>();
        private final List<String> outputNames = new ArrayList<>();
        private final List<Integer> outputLabelIndexes = new ArrayList<>();
        private final List<Integer> outputNumClasses = new ArrayList<>();
        
        public Builder(int batchSize) {
            this.batchSize = batchSize;
        }
        
        public Builder addReader(String name, RecordReader reader) {
            readers.add(reader);
            readerNames.add(name);
            return this;
        }
        
        public Builder addInput(String readerName) {
            inputNames.add(readerName);
            return this;
        }
        
        public Builder addOutputOneHot(String readerName, int labelIndex, int numClasses) {
            outputNames.add(readerName);
            outputLabelIndexes.add(labelIndex);
            outputNumClasses.add(numClasses);
            return this;
        }
        
        public DataSetIterator build() {
            // Pour simplifier, nous supposons que nous n'avons qu'un seul input et un seul output
            if (inputNames.size() != 1 || outputNames.size() != 1) {
                throw new IllegalArgumentException("Cette implémentation ne supporte qu'un seul input et un seul output");
            }
            
            String inputName = inputNames.get(0);
            String outputName = outputNames.get(0);
            int inputIdx = readerNames.indexOf(inputName);
            int outputIdx = readerNames.indexOf(outputName);
            
            if (inputIdx == -1 || outputIdx == -1) {
                throw new IllegalArgumentException("Noms de reader invalides");
            }
            
            RecordReader inputReader = readers.get(inputIdx);
            RecordReader outputReader = readers.get(outputIdx);
            
            int labelIndex = outputLabelIndexes.get(0);
            int numClasses = outputNumClasses.get(0);
            
            // Créer un DataSetIterator en utilisant les RecordReaders
            return new RecordReaderDataSetIterator(inputReader, outputReader, batchSize, labelIndex, numClasses);
        }
    }
}