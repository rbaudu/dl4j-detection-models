package com.project.models.sound;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.sound.sampled.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Modèle pour la détection de sons basé sur des spectrogrammes.
 * Convertit les fichiers audio en spectrogrammes et utilise un modèle VGG16 ou ResNet
 * pour classifier les sons.
 */
public class SpectrogramSoundModel {
    private static final Logger log = LoggerFactory.getLogger(SpectrogramSoundModel.class);
    
    // Paramètres pour l'extraction de spectrogrammes
    private final int sampleRate;
    private final int fftSize;
    private final int hopSize;
    private final int melBands;
    private final float minFreq;
    private final float maxFreq;
    
    // Paramètres du modèle
    private final Properties config;
    private final int inputHeight;
    private final int inputWidth;
    private final int channels;
    private final int numClasses;
    private final double learningRate;
    private final double dropoutRate;
    private final boolean useVGG16; // true pour VGG16, false pour ResNet
    private ComputationGraph network;
    
    // Mapping des classes
    private Map<String, Integer> classMapping = new HashMap<>();
    private List<String> classNames = new ArrayList<>();
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public SpectrogramSoundModel(Properties config) {
        this.config = config;
        
        // Paramètres du modèle
        this.inputHeight = Integer.parseInt(config.getProperty("sound.spectrogram.height", "224"));
        this.inputWidth = Integer.parseInt(config.getProperty("sound.spectrogram.width", "224"));
        this.channels = 3; // RGB
        this.numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "8"));
        this.learningRate = Double.parseDouble(config.getProperty("sound.model.learning.rate", "0.0001"));
        this.dropoutRate = Double.parseDouble(config.getProperty("sound.model.dropout", "0.5"));
        this.useVGG16 = Boolean.parseBoolean(config.getProperty("sound.model.use.vgg16", "true"));
        
        // Paramètres de l'extraction de spectrogrammes
        this.sampleRate = Integer.parseInt(config.getProperty("sound.sample.rate", "44100"));
        this.fftSize = Integer.parseInt(config.getProperty("sound.fft.size", "2048"));
        this.hopSize = Integer.parseInt(config.getProperty("sound.hop.size", "512"));
        this.melBands = Integer.parseInt(config.getProperty("sound.mel.bands", "128"));
        this.minFreq = Float.parseFloat(config.getProperty("sound.min.freq", "20"));
        this.maxFreq = Float.parseFloat(config.getProperty("sound.max.freq", "20000"));
    }
    
    /**
     * Initialise un nouveau modèle de détection de sons.
     * Charge soit VGG16 soit ResNet50 selon la configuration.
     * 
     * @throws IOException Si une erreur survient lors de l'initialisation
     */
    public void initNewModel() throws IOException {
        log.info("Initialisation d'un nouveau modèle pour la détection de sons basé sur spectrogrammes");
        
        // Charger le modèle pré-entraîné (VGG16 ou ResNet50)
        ZooModel<?> zooModel;
        String lastLayerName;
        int featureSize;
        
        if (useVGG16) {
            log.info("Utilisation de VGG16 pour l'analyse de spectrogrammes");
            zooModel = VGG16.builder()
                    .inputShape(new int[]{channels, inputHeight, inputWidth})
                    .build();
            lastLayerName = "fc2";
            featureSize = 4096;
        } else {
            log.info("Utilisation de ResNet50 pour l'analyse de spectrogrammes");
            zooModel = org.deeplearning4j.zoo.model.ResNet50.builder()
                    .inputShape(new int[]{channels, inputHeight, inputWidth})
                    .build();
            lastLayerName = "flatten";
            featureSize = 2048;
        }
        
        // Charger les poids pré-entraînés sur ImageNet
        ComputationGraph pretrained = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        
        // Configuration pour le fine-tuning
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(learningRate))
                .seed(123)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .dropOut(dropoutRate)
                .build();
        
        // Créer le modèle pour le transfert d'apprentissage
        network = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(lastLayerName)  // Geler jusqu'à cette couche
                .removeVertexAndConnections("predictions")  // Supprimer la couche de sortie existante
                .addLayer("predictions", 
                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(featureSize)
                                .nOut(numClasses)
                                .activation(Activation.SOFTMAX)
                                .build(), 
                        lastLayerName)  // Connecter à la dernière couche du feature extractor
                .setOutputs("predictions")
                .build();
        
        log.info("Modèle audio basé sur spectrogrammes initialisé avec succès");
    }
    
    /**
     * Charge le modèle par défaut spécifié dans la configuration.
     * S'il n'existe pas, initialise un nouveau modèle.
     *
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = config.getProperty("sound.spectrogram.model.path", "models/sound/spectrogram_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle de détection de sons basé sur spectrogrammes depuis {}", modelPath);
            loadModel(modelPath);
        } else {
            log.warn("Modèle par défaut non trouvé à {}, initialisation d'un nouveau modèle", modelPath);
            initNewModel();
            
            // Créer le répertoire parent
            File parentDir = modelFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }
            
            // Sauvegarder le modèle nouvellement créé
            saveModel(modelPath);
        }
    }
    
    /**
     * Charge un modèle existant à partir d'un fichier.
     *
     * @param modelPath Chemin du fichier modèle
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle de détection de sons basé sur spectrogrammes depuis {}", modelPath);
        
        try {
            network = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(modelPath);
            
            // Charger également le mapping des classes si disponible
            String mappingPath = modelPath.replace(".zip", "_classes.txt");
            File mappingFile = new File(mappingPath);
            if (mappingFile.exists()) {
                loadClassMapping(mappingFile);
            }
            
            log.info("Modèle et mapping des classes chargés avec succès");
        } catch (Exception e) {
            throw new IOException("Impossible de charger le modèle", e);
        }
    }
    
    /**
     * Charge le mapping des classes à partir d'un fichier.
     *
     * @param mappingFile Fichier contenant le mapping des classes
     * @throws IOException Si une erreur survient lors du chargement
     */
    private void loadClassMapping(File mappingFile) throws IOException {
        classNames.clear();
        classMapping.clear();
        
        List<String> lines = Files.readAllLines(mappingFile.toPath());
        for (int i = 0; i < lines.size(); i++) {
            String className = lines.get(i).trim();
            classNames.add(className);
            classMapping.put(className, i);
        }
        
        log.info("Chargé {} classes de sons", classNames.size());
    }
    
    /**
     * Sauvegarde le modèle et le mapping des classes dans un fichier.
     *
     * @param modelPath Chemin où sauvegarder le modèle
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public void saveModel(String modelPath) throws IOException {
        log.info("Sauvegarde du modèle de détection de sons basé sur spectrogrammes vers {}", modelPath);
        
        // Créer le répertoire parent si nécessaire
        File modelFile = new File(modelPath);
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        if (network != null) {
            org.deeplearning4j.util.ModelSerializer.writeModel(network, modelFile, true);
            
            // Sauvegarder également le mapping des classes
            if (!classNames.isEmpty()) {
                String mappingPath = modelPath.replace(".zip", "_classes.txt");
                Files.write(Paths.get(mappingPath), classNames);
            }
            
            log.info("Modèle et mapping des classes sauvegardés avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle à sauvegarder");
        }
    }
    
    /**
     * Exporte le modèle au format DL4J.
     *
     * @param exportPath Chemin où exporter le modèle
     * @throws IOException Si une erreur survient lors de l'export
     */
    public void exportModel(String exportPath) throws IOException {
        log.info("Exportation du modèle de détection de sons basé sur spectrogrammes vers {}", exportPath);
        
        boolean includeUpdater = Boolean.parseBoolean(config.getProperty("export.model.include.updater", "false"));
        
        if (network != null) {
            File modelFile = new File(exportPath);
            org.deeplearning4j.util.ModelSerializer.writeModel(network, modelFile, includeUpdater);
            
            // Exporter également le mapping des classes
            if (!classNames.isEmpty()) {
                String mappingPath = exportPath.replace(".zip", "_classes.txt");
                Files.write(Paths.get(mappingPath), classNames);
            }
            
            log.info("Modèle et mapping des classes exportés avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle à exporter");
        }
    }
    
    /**
     * Entraîne le modèle sur un ensemble de fichiers audio organisés par répertoires de classes.
     *
     * @param datasetPath Chemin vers le répertoire contenant les sous-répertoires de classes
     * @param epochs Nombre d'époques d'entraînement
     * @param batchSize Taille des batchs
     * @throws IOException Si une erreur survient lors de l'entraînement
     */
    public void trainOnDataset(String datasetPath, int epochs, int batchSize) throws IOException {
        log.info("Entraînement du modèle sur le jeu de données audio {}", datasetPath);
        
        File datasetDir = new File(datasetPath);
        if (!datasetDir.exists() || !datasetDir.isDirectory()) {
            throw new IllegalArgumentException("Le chemin spécifié n'est pas un répertoire valide");
        }
        
        // Scanners les sous-répertoires pour obtenir les classes
        File[] classDirs = datasetDir.listFiles(File::isDirectory);
        if (classDirs == null || classDirs.length == 0) {
            throw new IllegalArgumentException("Aucun sous-répertoire de classe trouvé");
        }
        
        // Créer le mapping des classes
        classNames.clear();
        classMapping.clear();
        for (int i = 0; i < classDirs.length; i++) {
            String className = classDirs[i].getName();
            classNames.add(className);
            classMapping.put(className, i);
        }
        
        log.info("Détecté {} classes de sons: {}", classNames.size(), String.join(", ", classNames));
        
        // Si le nombre de classes ne correspond pas à celui configuré, réinitialiser le modèle
        if (classNames.size() != numClasses) {
            log.warn("Le nombre de classes ({}) ne correspond pas à celui configuré ({}). Réinitialisation du modèle.",
                     classNames.size(), numClasses);
            initNewModel();
        }
        
        // TODO: Implémenter l'entraînement complet avec DataSetIterator
        // Pour cet exemple, nous montrons simplement comment générer les spectrogrammes
        log.info("Génération des spectrogrammes pour l'entraînement");
        
        // Parcourir chaque classe et générer des spectrogrammes
        for (File classDir : classDirs) {
            String className = classDir.getName();
            int classIndex = classMapping.get(className);
            
            File[] audioFiles = classDir.listFiles(file -> 
                    file.isFile() && (file.getName().endsWith(".wav") || 
                                     file.getName().endsWith(".mp3") ||
                                     file.getName().endsWith(".ogg")));
            
            if (audioFiles == null || audioFiles.length == 0) {
                log.warn("Aucun fichier audio trouvé dans la classe {}", className);
                continue;
            }
            
            log.info("Traitement de {} fichiers audio pour la classe {}", audioFiles.length, className);
            
            // Générer des spectrogrammes pour cette classe
            // Dans un cas réel, nous utiliserions ces spectrogrammes pour l'entraînement
            for (File audioFile : audioFiles) {
                try {
                    // Générer le spectrogramme
                    BufferedImage spectrogram = generateSpectrogram(audioFile.getPath());
                    
                    // Dans un cas réel, on sauvegarderait le spectrogramme ou on l'utiliserait directement
                    // pour l'entraînement. Ici, nous le générons simplement pour démonstration.
                    
                    // Convertir en INDArray
                    NativeImageLoader loader = new NativeImageLoader(inputHeight, inputWidth, channels);
                    INDArray spectrogramArray = loader.asMatrix(spectrogram);
                    
                    // Prétraiter pour le modèle
                    DataNormalization scaler = new VGG16ImagePreProcessor();
                    scaler.transform(spectrogramArray);
                    
                    // Dans un cas réel, on utiliserait ce spectrogramArray pour l'entraînement
                    // Pour cet exemple, nous simulons juste la génération
                    
                } catch (Exception e) {
                    log.error("Erreur lors du traitement du fichier {}: {}", audioFile.getPath(), e.getMessage());
                }
            }
        }
        
        log.info("Génération des spectrogrammes terminée. L'entraînement réel serait effectué ici.");
        
        // Sauvegarder le modèle et le mapping de classes
        saveModel(config.getProperty("sound.spectrogram.model.path", "models/sound/spectrogram_model.zip"));
    }
    
    /**
     * Prédit la classe d'un fichier audio.
     *
     * @param audioFilePath Chemin vers le fichier audio
     * @return Nom de la classe prédite
     * @throws IOException Si une erreur survient lors de la prédiction
     */
    public String predict(String audioFilePath) throws IOException {
        log.info("Prédiction de la classe pour le fichier audio {}", audioFilePath);
        
        if (network == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé");
        }
        
        // Générer le spectrogramme
        BufferedImage spectrogram = generateSpectrogram(audioFilePath);
        
        // Convertir en INDArray
        NativeImageLoader loader = new NativeImageLoader(inputHeight, inputWidth, channels);
        INDArray spectrogramArray = loader.asMatrix(spectrogram);
        
        // Prétraiter pour le modèle
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(spectrogramArray);
        
        // Faire la prédiction
        INDArray output = network.outputSingle(spectrogramArray);
        
        // Obtenir l'index de la classe avec la probabilité la plus élevée
        int predictedClass = output.argMax(1).getInt(0);
        
        // Convertir en nom de classe
        String className = "unknown";
        if (predictedClass >= 0 && predictedClass < classNames.size()) {
            className = classNames.get(predictedClass);
        }
        
        log.info("Classe prédite pour {}: {} (indice: {})", audioFilePath, className, predictedClass);
        return className;
    }
    
    /**
     * Génère un spectrogramme à partir d'un fichier audio.
     *
     * @param audioFilePath Chemin vers le fichier audio
     * @return Image du spectrogramme
     * @throws IOException Si une erreur survient lors de la génération
     */
    public BufferedImage generateSpectrogram(String audioFilePath) throws IOException {
        log.info("Génération de spectrogramme pour {}", audioFilePath);
        
        try {
            // Utiliser FFmpeg pour lire le fichier audio (supporte .wav, .mp3, .ogg, etc.)
            FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(audioFilePath);
            grabber.start();
            
            // Assurer que nous avons un canal audio
            if (grabber.getAudioChannels() < 1) {
                throw new IOException("Le fichier ne contient pas de données audio");
            }
            
            // Lire toutes les données audio
            List<Float> audioSamples = new ArrayList<>();
            Frame audioFrame;
            while ((audioFrame = grabber.grabSamples()) != null) {
                if (audioFrame.samples == null || audioFrame.samples.length == 0) {
                    continue;
                }
                
                // Convertir les échantillons en float
                FloatBuffer floatBuffer = null;
                
                if (audioFrame.samples[0] instanceof FloatBuffer) {
                    floatBuffer = (FloatBuffer) audioFrame.samples[0];
                } else if (audioFrame.samples[0] instanceof ShortBuffer) {
                    ShortBuffer shortBuffer = (ShortBuffer) audioFrame.samples[0];
                    float[] floatArray = new float[shortBuffer.limit()];
                    for (int i = 0; i < floatArray.length; i++) {
                        floatArray[i] = shortBuffer.get(i) / 32768.0f;
                    }
                    floatBuffer = FloatBuffer.wrap(floatArray);
                }
                
                if (floatBuffer != null) {
                    while (floatBuffer.hasRemaining()) {
                        audioSamples.add(floatBuffer.get());
                    }
                }
            }
            
            grabber.stop();
            grabber.release();
            
            // Convertir la liste en tableau
            float[] samples = new float[audioSamples.size()];
            for (int i = 0; i < samples.length; i++) {
                samples[i] = audioSamples.get(i);
            }
            
            // Générer le spectrogramme Mel
            BufferedImage melSpectrogram = generateMelSpectrogram(samples, sampleRate, fftSize, hopSize, melBands);
            
            return melSpectrogram;
            
        } catch (Exception e) {
            throw new IOException("Erreur lors de la génération du spectrogramme", e);
        }
    }
    
    /**
     * Génère un spectrogramme mel à partir des échantillons audio.
     *
     * @param samples Échantillons audio
     * @param sampleRate Taux d'échantillonnage
     * @param fftSize Taille de la FFT
     * @param hopSize Taille du hop
     * @param melBands Nombre de bandes mel
     * @return Image du spectrogramme
     */
    private BufferedImage generateMelSpectrogram(float[] samples, int sampleRate, int fftSize, int hopSize, int melBands) {
        int width = (samples.length / hopSize);
        int height = melBands;
        
        // Créer une matrice pour le spectrogramme mel
        float[][] melSpec = new float[height][width];
        
        // Calculer le spectrogramme mel (simulation pour cet exemple)
        // Dans un cas réel, nous utiliserions une bibliothèque comme librosa ou similaire
        // Ici, nous simulons simplement un spectrogramme
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int startSample = j * hopSize;
                if (startSample < samples.length - fftSize) {
                    // Simuler l'énergie dans cette bande de fréquence et ce frame temporel
                    float energy = 0;
                    for (int k = 0; k < fftSize; k++) {
                        energy += Math.abs(samples[startSample + k]);
                    }
                    melSpec[i][j] = energy / fftSize;
                }
            }
        }
        
        // Convertir en dB et normaliser
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // Convertir en dB
                melSpec[i][j] = (float) (10 * Math.log10(1e-10 + melSpec[i][j]));
                
                // Mettre à jour min/max
                min = Math.min(min, melSpec[i][j]);
                max = Math.max(max, melSpec[i][j]);
            }
        }
        
        // Normaliser
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                melSpec[i][j] = (melSpec[i][j] - min) / (max - min);
            }
        }
        
        // Créer l'image du spectrogramme
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float value = melSpec[height - 1 - i][j]; // inverser l'axe y pour que les basses fréquences soient en bas
                
                // Utiliser une palette de couleurs (ici, du noir au jaune en passant par le bleu et le violet)
                int r, g, b;
                
                if (value < 0.33) {
                    // Noir vers bleu
                    float factor = value / 0.33f;
                    r = 0;
                    g = 0;
                    b = (int) (255 * factor);
                } else if (value < 0.66) {
                    // Bleu vers violet/rose
                    float factor = (value - 0.33f) / 0.33f;
                    r = (int) (255 * factor);
                    g = 0;
                    b = 255;
                } else {
                    // Violet/rose vers jaune
                    float factor = (value - 0.66f) / 0.34f;
                    r = 255;
                    g = (int) (255 * factor);
                    b = 255 - (int) (255 * factor);
                }
                
                // Définir la couleur du pixel
                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(j, i, rgb);
            }
        }
        
        // Redimensionner l'image à la taille souhaitée pour le modèle
        BufferedImage resizedImage = new BufferedImage(inputWidth, inputHeight, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(image, 0, 0, inputWidth, inputHeight, null);
        
        return resizedImage;
    }
    
    /**
     * Obtient le réseau de neurones.
     *
     * @return Le réseau de neurones
     */
    public ComputationGraph getNetwork() {
        return network;
    }
}
