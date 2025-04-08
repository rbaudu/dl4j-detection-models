// src/test/java/com/project/test/SoundModelTester.java (version corrigée)
package com.project.test;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

/**
 * Classe de test pour le modèle de détection de son
 * Cette implémentation suppose que les caractéristiques audio ont été prétraitées
 * et stockées dans des fichiers CSV.
 */
public class SoundModelTester extends AbstractModelTester {
    
    private static final Logger log = LoggerFactory.getLogger(SoundModelTester.class);
    
    private final int batchSize;
    private final int numFeatures;
    private final Random rng;
    
    public SoundModelTester(Properties config) {
        super(config);
        
        // Configurer les chemins de fichiers
        this.modelPath = config.getProperty("sound.model.dir") + "/" + 
                config.getProperty("sound.model.name") + ".zip";
        this.testDataPath = config.getProperty("sound.test.data.dir", 
                config.getProperty("sound.data.dir") + "/test");
        
        // Configurer les paramètres
        this.batchSize = Integer.parseInt(config.getProperty("sound.model.batch.size", "32"));
        this.numFeatures = Integer.parseInt(config.getProperty("sound.model.input.size", "256"));
        this.rng = new Random(Integer.parseInt(config.getProperty("training.seed", "123")));
    }
    
    @Override
    protected DataSetIterator createTestDataIterator() throws IOException {
        // S'assurer que le répertoire de test existe
        File testDir = new File(testDataPath);
        if (!testDir.exists()) {
            log.error("Le répertoire de données de test n'existe pas: {}", testDataPath);
            throw new IOException("Répertoire de test non trouvé: " + testDataPath);
        }
        
        // Nous supposons que les caractéristiques audio sont déjà extraites et stockées dans des fichiers CSV
        // Format attendu: chaque ligne = vecteur de caractéristiques, dernière colonne = étiquette de classe
        FileSplit testData = new FileSplit(testDir, new String[]{".csv"}, rng);
        
        try {
            // Créer un lecteur CSV
            RecordReader recordReader = new CSVRecordReader();
            recordReader.initialize(testData);
            
            // Déterminer le nombre de classes
            int numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "2"));
            
            // Créer le DataSetIterator
            DataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                    .classification(numFeatures, numClasses)
                    .build();
            
            // Configurer la normalisation des données
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(testIterator);
            testIterator.setPreProcessor(normalizer);
            
            log.info("Iterator de test créé avec {} caractéristiques et {} classes", 
                    numFeatures, numClasses);
            return testIterator;
            
        } catch (InterruptedException e) {
            throw new IOException("Erreur lors de l'initialisation du lecteur de données", e);
        }
    }
}