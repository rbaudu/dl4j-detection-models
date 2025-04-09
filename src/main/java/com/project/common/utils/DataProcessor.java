package com.project.common.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DataProcessor {

    /**
     * Divise un ensemble de données en ensembles d'entraînement et de test
     * @param dataset L'ensemble de données complet
     * @param trainRatio Proportion des données pour l'entraînement (0.0-1.0)
     * @return Un tableau de deux éléments: [données d'entraînement, données de test]
     */
    public static DataSet[] splitDataSet(DataSet dataset, double trainRatio) {
        int numExamples = dataset.numExamples();
        int trainSize = (int) (numExamples * trainRatio);
        int testSize = numExamples - trainSize;
        
        // Mélanger les données
        dataset.shuffle();
        
        // Créer les ensembles d'entraînement et de test
        DataSet trainSet = new DataSet(
                dataset.getFeatures().get(NDArrayIndex.interval(0, trainSize)),
                dataset.getLabels().get(NDArrayIndex.interval(0, trainSize))
        );
        
        DataSet testSet = new DataSet(
                dataset.getFeatures().get(NDArrayIndex.interval(trainSize, numExamples)),
                dataset.getLabels().get(NDArrayIndex.interval(trainSize, numExamples))
        );
        
        // Retourner les deux ensembles de données
        return new DataSet[] { trainSet, testSet };
    }
    
    /**
     * Normalise les caractéristiques d'un ensemble de données
     * @param dataset L'ensemble de données à normaliser
     * @return L'ensemble de données normalisé
     */
    public static DataSet normalizeDataSet(DataSet dataset) {
        INDArray features = dataset.getFeatures();
        INDArray mean = features.mean(0);
        INDArray std = features.std(0);
        
        // Éviter la division par zéro
        for (int i = 0; i < std.length(); i++) {
            if (std.getDouble(i) < 1e-6) {
                std.putScalar(i, 1.0);
            }
        }
        
        // Normaliser les caractéristiques (z-score)
        features.subiRowVector(mean).diviRowVector(std);
        
        return dataset;
    }
    
    /**
     * Combine plusieurs ensembles de données en un seul
     * @param dataSets Liste d'ensembles de données à combiner
     * @return Un ensemble de données combiné
     */
    public static DataSet combineDataSets(List<DataSet> dataSets) {
        if (dataSets == null || dataSets.isEmpty()) {
            throw new IllegalArgumentException("La liste d'ensembles de données ne peut pas être vide");
        }
        
        if (dataSets.size() == 1) {
            return dataSets.get(0);
        }
        
        // Créer une liste pour contenir tous les exemples
        List<DataSet> allExamples = new ArrayList<>();
        
        // Extraire tous les exemples individuels
        for (DataSet dataSet : dataSets) {
            for (int i = 0; i < dataSet.numExamples(); i++) {
                allExamples.add(dataSet.get(i));
            }
        }
        
        // Combiner tous les exemples en un seul ensemble de données
        return DataSet.merge(allExamples);
    }
    
    /**
     * Augmente un ensemble de données avec du bruit gaussien
     * @param dataset L'ensemble de données à augmenter
     * @param noiseLevel Niveau de bruit à ajouter (écart-type)
     * @return L'ensemble de données augmenté
     */
    public static DataSet augmentWithNoise(DataSet dataset, double noiseLevel) {
        INDArray features = dataset.getFeatures().dup();
        INDArray labels = dataset.getLabels().dup();
        
        // Créer du bruit gaussien avec la même forme que les caractéristiques
        int[] shape = features.shape();
        
        // Correction de l'utilisation de Nd4j.rand()
        Distribution dist = new NormalDistribution(0, noiseLevel);
        INDArray noise = Nd4j.rand(DataType.FLOAT, shape, dist);
        
        // Ajouter le bruit aux caractéristiques
        features.addi(noise);
        
        // Créer un nouvel ensemble de données avec les caractéristiques bruitées
        return new DataSet(features, labels);
    }
}