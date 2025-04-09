package com.project.training;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import javax.sound.sampled.UnsupportedAudioFileException;

import com.project.common.utils.AudioUtils;
import com.project.common.utils.ModelUtils;

public class SoundTrainer extends ModelTrainer {

    private int featureSize;
    private int numClasses;
    private Random rng;

    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param featureSize Taille des caractéristiques audio
     * @param numClasses Nombre de classes de sortie
     */
    public SoundTrainer(int batchSize, int numEpochs, String modelOutputPath, int featureSize, int numClasses) {
        super(batchSize, numEpochs, modelOutputPath);
        this.featureSize = featureSize;
        this.numClasses = numClasses;
        this.rng = new Random(42);
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public SoundTrainer(Properties config) {
        super(config);
        this.featureSize = Integer.parseInt(config.getProperty("sound.feature.size", "128"));
        this.numClasses = Integer.parseInt(config.getProperty("sound.num.classes", "3"));
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
        return splitDataset(allData, trainTestRatio);
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

        // Créer un DataSetIterator pour chaque lecteur
        DataSetIterator featuresIterator = new RecordReaderDataSetIterator(featuresReader, batchSize);
        DataSetIterator labelsIterator = new RecordReaderDataSetIterator(labelsReader, batchSize, 0, numClasses);

        // Collecter toutes les données
        List<DataSet> featureDatasets = new ArrayList<>();
        List<DataSet> labelDatasets = new ArrayList<>();
        
        while (featuresIterator.hasNext() && labelsIterator.hasNext()) {
            featureDatasets.add(featuresIterator.next());
            labelDatasets.add(labelsIterator.next());
        }
        
        // Fusionner les caractéristiques et les étiquettes
        DataSet allFeatures = DataSet.merge(featureDatasets);
        DataSet allLabels = DataSet.merge(labelDatasets);
        
        // Créer un DataSet complet
        DataSet allData = new DataSet(allFeatures.getFeatures(), allLabels.getLabels());
        allData.shuffle();

        // Diviser en ensembles d'entraînement et de test
        return splitDataset(allData, trainTestRatio);
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
    
    @Override
    protected DataSet prepareData() throws IOException {
        try {
            // Utiliser le répertoire spécifié dans la configuration
            String audioDir = config.getProperty("sound.data.dir", "data/audio");
            
            // Préparer les données audio
            DataSet[] datasets = prepareAudioData(audioDir, 0.8);
            
            // Retourner l'ensemble complet
            return DataSet.merge(Arrays.asList(datasets));
        } catch (UnsupportedAudioFileException e) {
            throw new IOException("Erreur lors du traitement des fichiers audio", e);
        }
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        MultiLayerNetwork network = ModelUtils.createSimpleNetwork(featureSize, numClasses);
        return network;
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = config != null ? 
            config.getProperty("sound.model.path", modelOutputPath) : 
            modelOutputPath;
        
        ModelUtils.saveModel(network, modelPath, true);
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        // Déterminer le chemin du checkpoint
        String baseDir = config != null ? 
            config.getProperty("sound.checkpoint.dir", "checkpoints/sound") : 
            "checkpoints/sound";
        
        // Assurer que le répertoire existe
        createDirectories(baseDir);
        
        // Créer le chemin complet
        String checkpointPath = baseDir + "/sound_model_epoch_" + epoch + ".zip";
        
        // Sauvegarder le checkpoint
        ModelUtils.saveModel(network, checkpointPath, true);
    }
}