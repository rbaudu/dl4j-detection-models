package com.project.training;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ActivityTrainer extends ModelTrainer {

    private int height;
    private int width;
    private int channels;
    private int numClasses;
    private Random rng;

    public ActivityTrainer(int batchSize, int numEpochs, String modelOutputPath, int height, int width, int channels, int numClasses) {
        super(batchSize, numEpochs, modelOutputPath);
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numClasses = numClasses;
        this.rng = new Random(42);
    }

    /**
     * Prépare et charge les données d'images pour l'entrainement
     * @param dataDir Répertoire contenant les images d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareData(String dataDir, double trainTestRatio) throws IOException {
        // Créer un FileSplit pour le répertoire de données
        File dataDirFile = new File(dataDir);
        FileSplit dataSplit = new FileSplit(dataDirFile, NativeImageLoader.ALLOWED_FORMATS, rng);

        // Créer un ImageRecordReader pour lire les images
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, dataSplit.getRootDir());
        recordReader.initialize(dataSplit);

        // Créer un DataSetIterator pour convertir les images en DataSet
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);

        // Normaliser les données d'image (mettre à l'échelle les pixels entre 0 et 1)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

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
     * Prépare et charge les données d'images avec augmentation de données
     * @param dataDir Répertoire contenant les images d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @return Un tableau de deux éléments: [trainData, testData]
     */
    public DataSet[] prepareDataWithAugmentation(String dataDir, double trainTestRatio) throws IOException {
        // Créer un FileSplit pour le répertoire de données
        File dataDirFile = new File(dataDir);
        FileSplit dataSplit = new FileSplit(dataDirFile, NativeImageLoader.ALLOWED_FORMATS, rng);

        // Définir des transformations pour l'augmentation de données
        ImageTransform resizeTransform = new ResizeImageTransform(height, width);
        
        // Créer une liste de transformations
        List<Pair<ImageTransform, Double>> transforms = new ArrayList<>();
        transforms.add(new Pair<>(resizeTransform, 1.0));
        
        // Créer un pipeline de transformations
        PipelineImageTransform pipeline = new PipelineImageTransform(transforms, false);
        
        // Créer un ImageRecordReader pour lire les images
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, dataSplit.getRootDir());
        recordReader.initialize(dataSplit, pipeline);

        // Créer un DataSetIterator pour convertir les images en DataSet
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);

        // Normaliser les données d'image (mettre à l'échelle les pixels entre 0 et 1)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

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
     * Initialise et entraîne le modèle pour la classification d'activité
     * @param dataDir Répertoire contenant les images d'entraînement
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnActivityData(String dataDir, double trainTestRatio) throws IOException {
        // Préparer les données
        DataSet[] data = prepareData(dataDir, trainTestRatio);
        DataSet trainData = data[0];
        DataSet testData = data[1];

        // Initialiser le modèle
        int numInputs = height * width * channels;
        initializeModel(numInputs, numClasses);

        // Entraîner le modèle
        train(trainData, testData);
    }
}