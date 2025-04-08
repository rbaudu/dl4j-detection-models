package com.project.training;

import com.project.common.utils.AudioUtils;
import com.project.common.utils.DataProcessor;
import com.project.models.sound.SoundModel;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

/**
 * Classe pour l'entraînement du modèle de détection de sons.
 * Implémente les méthodes spécifiques pour préparer les données et gérer le modèle de sons.
 * Utilise le transfert d'apprentissage avec YAMNet.
 */
public class SoundTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(SoundTrainer.class);
    
    private final SoundModel model;
    private final String dataDir;
    private final String modelDir;
    private final String modelName;
    private final Random random;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public SoundTrainer(Properties config) {
        super(config);
        this.model = new SoundModel(config);
        this.dataDir = config.getProperty("sound.data.dir", "data/raw/sound");
        this.modelDir = config.getProperty("sound.model.dir", "models/sound");
        this.modelName = config.getProperty("sound.model.name", "sound_model");
        
        // Initialiser le générateur de nombres aléatoires
        int seed = Integer.parseInt(config.getProperty("training.seed", "123"));
        this.random = new Random(seed);
        
        try {
            createDirectories(modelDir);
        } catch (IOException e) {
            log.error("Erreur lors de la création des répertoires", e);
        }
    }
    
    @Override
    protected DataSet prepareData() throws IOException {
        log.info("Préparation des données pour le modèle de détection de sons");
        
        // Vérifier si le répertoire de données existe
        File dataDirFile = new File(dataDir);
        if (!dataDirFile.exists() || !dataDirFile.isDirectory()) {
            log.error("Le répertoire de données n'existe pas: {}", dataDir);
            return generateSyntheticData();
        }
        
        // Lister les fichiers audio dans le répertoire
        List<File> audioFiles = new ArrayList<>();
        File[] files = dataDirFile.listFiles();
        
        if (files != null) {
            for (File file : files) {
                if (AudioUtils.isAudioFile(file)) {
                    audioFiles.add(file);
                }
            }
        }
        
        // Si aucun fichier audio n'a été trouvé, générer des exemples synthétiques
        if (audioFiles.isEmpty()) {
            log.warn("Aucun fichier audio trouvé dans {}. Utilisation de données synthétiques", dataDir);
            return generateSyntheticData();
        }
        
        log.info("Chargement de {} fichiers audio pour l'entraînement", audioFiles.size());
        
        // Prétraiter les fichiers audio pour YAMNet
        List<double[]> features = new ArrayList<>();
        List<double[]> labels = new ArrayList<>();
        
        int numClasses = model.getNumSoundClasses();
        
        // Traiter chaque fichier audio
        for (File audioFile : audioFiles) {
            try {
                // Prétraitement du fichier audio
                INDArray audioFeatures = AudioUtils.preprocessAudioForYAMNet(audioFile);
                
                // Déterminer la classe depuis le nom du fichier ou le répertoire parent
                String filename = audioFile.getName().toLowerCase();
                String parentDir = audioFile.getParentFile().getName().toLowerCase();
                
                // Ici, nous déterminons la classe approximativement à partir du nom de fichier
                // Dans un cas réel, vous auriez besoin d'une stratégie plus précise
                int classIdx = 0; // Par défaut "Silence"
                
                if (filename.contains("speech") || filename.contains("voice") || 
                    filename.contains("talk") || parentDir.contains("speech")) {
                    classIdx = 1; // "Parole"
                } else if (filename.contains("music") || parentDir.contains("music")) {
                    classIdx = 2; // "Musique"
                } else if (filename.contains("ambient") || filename.contains("background") || 
                           parentDir.contains("ambient")) {
                    classIdx = 3; // "Bruit ambiant"
                } else if (filename.contains("alarm") || filename.contains("alert") || 
                           parentDir.contains("alarm")) {
                    classIdx = 4; // "Alarme"
                }
                
                // Créer l'étiquette one-hot
                double[] label = new double[numClasses];
                label[classIdx] = 1.0;
                
                // Ajouter aux listes
                features.add(audioFeatures.toDoubleVector());
                labels.add(label);
                
            } catch (IOException e) {
                log.warn("Impossible de traiter le fichier audio {}: {}", audioFile.getName(), e.getMessage());
            }
        }
        
        // Si aucun fichier n'a pu être traité, utiliser des données synthétiques
        if (features.isEmpty()) {
            log.warn("Aucun fichier audio n'a pu être traité. Utilisation de données synthétiques");
            return generateSyntheticData();
        }
        
        // Convertir les listes en DataSet
        try {
            // Créer un RecordReader à partir des collections
            RecordReader featuresReader = new CollectionRecordReader(features);
            RecordReader labelsReader = new CollectionRecordReader(labels);
            
            // Créer les itérateurs
            DataSetIterator iterator = new RecordReaderDataSetIterator(featuresReader, labelsReader, batchSize);
            
            // Combiner tous les batchs en un seul DataSet
            DataSet allData = new DataSet();
            while (iterator.hasNext()) {
                allData.add(iterator.next());
            }
            
            log.info("Données préparées: {} exemples", allData.numExamples());
            
            return allData;
            
        } catch (Exception e) {
            log.error("Erreur lors de la création du DataSet", e);
            return generateSyntheticData();
        }
    }
    
    /**
     * Génère des données audio synthétiques pour l'entraînement.
     * Cette méthode est utilisée uniquement lorsqu'aucune donnée réelle n'est disponible.
     *
     * @return DataSet contenant les données synthétiques
     */
    private DataSet generateSyntheticData() {
        log.info("Génération de données audio synthétiques");
        
        int numExamples = 100; // 20 exemples par classe (pour 5 classes)
        int sampleLength = AudioUtils.YAMNET_WAVEFORM_LENGTH;
        int numClasses = model.getNumSoundClasses();
        
        return DataProcessor.createSyntheticAudioDataSet(numExamples, sampleLength, numClasses, random);
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        // Initialiser un nouveau modèle pour le transfert d'apprentissage
        model.initNewModel();
        return model.getNetwork();
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        log.info("Sauvegarde du modèle final vers {}", modelPath);
        model.saveModel(modelPath);
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        String checkpointPath = new File(modelDir + "/checkpoints", 
                                         modelName + "_epoch_" + epoch + ".zip").getPath();
        log.info("Sauvegarde du checkpoint à l'époque {} vers {}", epoch, checkpointPath);
        model.getNetwork().save(new File(checkpointPath), true);
    }
}
