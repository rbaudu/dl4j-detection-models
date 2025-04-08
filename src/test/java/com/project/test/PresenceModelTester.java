package com.project.test;

import com.project.common.utils.DataProcessor;
import com.project.common.utils.ImageUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

/**
 * Classe de test pour le modèle de détection de présence
 */
public class PresenceModelTester extends AbstractModelTester {
    
    private static final Logger log = LoggerFactory.getLogger(PresenceModelTester.class);
    
    private final int batchSize;
    private final int imageSize;
    private final Random rng;
    
    public PresenceModelTester(Properties config) {
        super(config);
        
        // Configurer les chemins de fichiers
        this.modelPath = config.getProperty("presence.model.dir") + "/" + 
                config.getProperty("presence.model.name") + ".zip";
        this.testDataPath = config.getProperty("presence.test.data.dir", 
                config.getProperty("presence.data.dir") + "/test");
        
        // Configurer les paramètres
        this.batchSize = Integer.parseInt(config.getProperty("presence.model.batch.size", "32"));
        this.imageSize = Integer.parseInt(config.getProperty("presence.model.input.size", "64"));
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
        
        // Configurer le traitement des images
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        // Créer le RecordReader
        ImageRecordReader recordReader = new ImageRecordReader(imageSize, imageSize, 3, labelMaker);
        ImageTransform resize = new ResizeImageTransform(imageSize, imageSize);
        recordReader.initialize(testData, resize);
        
        // Créer le DataSetIterator
        DataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                .classification(1, recordReader.numLabels())
                .build();
        
        // Configurer la normalisation des données
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(testIterator);
        testIterator.setPreProcessor(scaler);
        
        log.info("Iterator de test créé avec {} classes", recordReader.numLabels());
        return testIterator;
    }
}
