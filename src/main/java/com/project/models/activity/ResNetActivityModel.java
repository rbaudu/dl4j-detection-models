package com.project.models.activity;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Modèle ResNet pour la détection d'activité.
 * Utilise le transfert d'apprentissage à partir d'un modèle ResNet50 pré-entraîné.
 */
public class ResNetActivityModel {
    private static final Logger log = LoggerFactory.getLogger(ResNetActivityModel.class);
    
    private final Properties config;
    private final int inputHeight;
    private final int inputWidth;
    private final int channels;
    private final int numClasses;
    private final double learningRate;
    private final double dropoutRate;
    private ComputationGraph resNetNetwork;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public ResNetActivityModel(Properties config) {
        this.config = config;
        this.inputHeight = Integer.parseInt(config.getProperty("activity.model.input.height", "224"));
        this.inputWidth = Integer.parseInt(config.getProperty("activity.model.input.width", "224"));
        this.channels = 3; // RGB
        this.numClasses = Integer.parseInt(config.getProperty("activity.model.num.classes", "27"));
        this.learningRate = Double.parseDouble(config.getProperty("activity.model.learning.rate", "0.0001"));
        this.dropoutRate = Double.parseDouble(config.getProperty("activity.model.dropout", "0.5"));
    }
    
    /**
     * Initialise un nouveau modèle ResNet pour la détection d'activité.
     * Charge le modèle ResNet50 pré-entraîné et le configure pour le transfert d'apprentissage.
     * 
     * @throws IOException Si une erreur survient lors du chargement des poids pré-entraînés
     */
    public void initNewModel() throws IOException {
        log.info("Initialisation d'un nouveau modèle ResNet pour la détection d'activité");
        
        // Charger le modèle ResNet50 pré-entraîné
        ZooModel<?> zooModel = ResNet50.builder()
                .inputShape(new int[]{channels, inputHeight, inputWidth})
                .build();
        
        // Utiliser des poids pré-entraînés sur ImageNet
        ComputationGraph pretrained = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info("Modèle ResNet50 pré-entraîné chargé avec succès");
        
        // Configuration pour le fine-tuning
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(learningRate))
                .seed(123)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .dropOut(dropoutRate)
                .build();
        
        // Pour ResNet50, la couche avant la sortie est généralement "flatten" ou "avg_pool"
        // La sortie de cette couche est de taille 2048
        resNetNetwork = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("flatten")  // Geler jusqu'à la couche flatten
                .removeVertexAndConnections("fc1000")  // Supprimer la couche de sortie existante
                .addLayer("fc1000", 
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2048)  // ResNet50 a 2048 caractéristiques en sortie de flatten
                                .nOut(numClasses)
                                .activation(Activation.SOFTMAX)
                                .build(), 
                        "flatten")  // Connecter à la couche flatten
                .setOutputs("fc1000")
                .build();
        
        log.info("Modèle ResNet pour la détection d'activité configuré avec succès");
    }
    
    /**
     * Charge le modèle par défaut spécifié dans la configuration.
     * S'il n'existe pas, initialise un nouveau modèle.
     *
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = config.getProperty("activity.resnet.model.path", "models/activity/resnet_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle ResNet de détection d'activité depuis {}", modelPath);
            loadModel(modelPath);
        } else {
            log.warn("Modèle ResNet par défaut non trouvé à {}, initialisation d'un nouveau modèle", modelPath);
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
     * Charge un modèle ResNet existant à partir d'un fichier.
     *
     * @param modelPath Chemin du fichier modèle
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle ResNet de détection d'activité depuis {}", modelPath);
        
        try {
            resNetNetwork = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(modelPath);
            log.info("Modèle ResNet chargé avec succès");
        } catch (Exception e) {
            throw new IOException("Impossible de charger le modèle ResNet", e);
        }
    }
    
    /**
     * Sauvegarde le modèle ResNet dans un fichier.
     *
     * @param modelPath Chemin où sauvegarder le modèle
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public void saveModel(String modelPath) throws IOException {
        log.info("Sauvegarde du modèle ResNet de détection d'activité vers {}", modelPath);
        
        // Créer le répertoire parent si nécessaire
        File modelFile = new File(modelPath);
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        if (resNetNetwork != null) {
            // Dans beta7, le paramètre s'appelle saveUpdater au lieu de includeUpdater
            org.deeplearning4j.util.ModelSerializer.writeModel(resNetNetwork, modelFile, true);
            log.info("Modèle ResNet sauvegardé avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle ResNet à sauvegarder");
        }
    }
    
    /**
     * Exporte le modèle ResNet au format DL4J.
     *
     * @param exportPath Chemin où exporter le modèle
     * @throws IOException Si une erreur survient lors de l'export
     */
    public void exportModel(String exportPath) throws IOException {
        log.info("Exportation du modèle ResNet de détection d'activité vers {}", exportPath);
        
        boolean saveUpdater = Boolean.parseBoolean(config.getProperty("export.model.include.updater", "false"));
        
        if (resNetNetwork != null) {
            File modelFile = new File(exportPath);
            // Dans beta7, le paramètre s'appelle saveUpdater au lieu de includeUpdater
            org.deeplearning4j.util.ModelSerializer.writeModel(resNetNetwork, modelFile, saveUpdater);
            log.info("Modèle ResNet exporté avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle ResNet à exporter");
        }
    }
    
    /**
     * Prédit l'activité à partir d'une image.
     * 
     * @param imageData Image sous forme de tableau de pixels RGB (hauteur x largeur x 3)
     * @return Index de la classe prédite
     */
    public int predictActivity(INDArray imageData) {
        if (resNetNetwork == null) {
            throw new IllegalStateException("Le modèle ResNet n'est pas initialisé");
        }
        
        // Prétraitement de l'image pour ResNet50
        // Utiliser le préprocesseur VGG16 car il applique la même normalisation que pour ResNet
        DataNormalization preProcessor = new VGG16ImagePreProcessor();
        preProcessor.transform(imageData);
        
        // Faire la prédiction
        INDArray output = resNetNetwork.outputSingle(imageData);
        
        // Obtenir l'index de la classe avec la probabilité la plus élevée
        return output.argMax(1).getInt(0);
    }
    
    /**
     * Obtient le réseau ResNet.
     *
     * @return Le réseau ResNet
     */
    public ComputationGraph getResNetNetwork() {
        return resNetNetwork;
    }
}