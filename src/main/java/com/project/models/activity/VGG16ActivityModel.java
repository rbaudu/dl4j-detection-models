package com.project.models.activity;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Modèle VGG16 pour la détection d'activité.
 * Utilise le transfert d'apprentissage à partir d'un modèle VGG16 pré-entraîné.
 */
public class VGG16ActivityModel {
    private static final Logger log = LoggerFactory.getLogger(VGG16ActivityModel.class);
    
    private final Properties config;
    private final int inputHeight;
    private final int inputWidth;
    private final int channels;
    private final int numClasses;
    private final double learningRate;
    private final double dropoutRate;
    private ComputationGraph vgg16Network;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public VGG16ActivityModel(Properties config) {
        this.config = config;
        this.inputHeight = Integer.parseInt(config.getProperty("activity.model.input.height", "224"));
        this.inputWidth = Integer.parseInt(config.getProperty("activity.model.input.width", "224"));
        this.channels = 3; // RGB
        this.numClasses = Integer.parseInt(config.getProperty("activity.model.num.classes", "27"));
        this.learningRate = Double.parseDouble(config.getProperty("activity.model.learning.rate", "0.0001"));
        this.dropoutRate = Double.parseDouble(config.getProperty("activity.model.dropout", "0.5"));
    }
    
    /**
     * Initialise un nouveau modèle VGG16 pour la détection d'activité.
     * Charge le modèle VGG16 pré-entraîné et le configure pour le transfert d'apprentissage.
     * 
     * @throws IOException Si une erreur survient lors du chargement des poids pré-entraînés
     */
    public void initNewModel() throws IOException {
        log.info("Initialisation d'un nouveau modèle VGG16 pour la détection d'activité");
        
        // Charger le modèle VGG16 pré-entraîné
        ZooModel<?> zooModel = VGG16.builder()
                .inputShape(new int[]{channels, inputHeight, inputWidth})
                .build();
        
        // Utiliser des poids pré-entraînés sur ImageNet
        ComputationGraph pretrained = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info("Modèle VGG16 pré-entraîné chargé avec succès");
        
        // Configuration pour le fine-tuning
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(learningRate))
                .seed(123)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .dropOut(dropoutRate)
                .build();
        
        // Créer le modèle pour le transfert d'apprentissage
        // Pour VGG16, la dernière couche de transfert est généralement "fc2"
        vgg16Network = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")  // Geler jusqu'à la couche fc2
                .removeVertexAndConnections("predictions")  // Supprimer la couche de prédiction existante
                .addLayer("predictions", 
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096)  // fc2 a 4096 sorties
                                .nOut(numClasses)
                                .activation(Activation.SOFTMAX)
                                .build(), 
                        "fc2")  // Connecter à la couche fc2
                .setOutputs("predictions")
                .build();
        
        log.info("Modèle VGG16 pour la détection d'activité configuré avec succès");
    }
    
    /**
     * Charge le modèle par défaut spécifié dans la configuration.
     * S'il n'existe pas, initialise un nouveau modèle.
     *
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadDefaultModel() throws IOException {
        String modelPath = config.getProperty("activity.vgg16.model.path", "models/activity/vgg16_model.zip");
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            log.info("Chargement du modèle VGG16 de détection d'activité depuis {}", modelPath);
            loadModel(modelPath);
        } else {
            log.warn("Modèle VGG16 par défaut non trouvé à {}, initialisation d'un nouveau modèle", modelPath);
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
     * Charge un modèle VGG16 existant à partir d'un fichier.
     *
     * @param modelPath Chemin du fichier modèle
     * @throws IOException Si une erreur survient lors du chargement
     */
    public void loadModel(String modelPath) throws IOException {
        log.info("Chargement du modèle VGG16 de détection d'activité depuis {}", modelPath);
        
        try {
            vgg16Network = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(modelPath);
            log.info("Modèle VGG16 chargé avec succès");
        } catch (Exception e) {
            throw new IOException("Impossible de charger le modèle VGG16", e);
        }
    }
    
    /**
     * Sauvegarde le modèle VGG16 dans un fichier.
     *
     * @param modelPath Chemin où sauvegarder le modèle
     * @throws IOException Si une erreur survient lors de la sauvegarde
     */
    public void saveModel(String modelPath) throws IOException {
        log.info("Sauvegarde du modèle VGG16 de détection d'activité vers {}", modelPath);
        
        // Créer le répertoire parent si nécessaire
        File modelFile = new File(modelPath);
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        if (vgg16Network != null) {
            org.deeplearning4j.util.ModelSerializer.writeModel(vgg16Network, modelFile, true);
            log.info("Modèle VGG16 sauvegardé avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle VGG16 à sauvegarder");
        }
    }
    
    /**
     * Exporte le modèle VGG16 au format DL4J.
     *
     * @param exportPath Chemin où exporter le modèle
     * @throws IOException Si une erreur survient lors de l'export
     */
    public void exportModel(String exportPath) throws IOException {
        log.info("Exportation du modèle VGG16 de détection d'activité vers {}", exportPath);
        
        boolean includeUpdater = Boolean.parseBoolean(config.getProperty("export.model.include.updater", "false"));
        
        if (vgg16Network != null) {
            File modelFile = new File(exportPath);
            org.deeplearning4j.util.ModelSerializer.writeModel(vgg16Network, modelFile, includeUpdater);
            log.info("Modèle VGG16 exporté avec succès");
        } else {
            throw new IllegalStateException("Aucun modèle VGG16 à exporter");
        }
    }
    
    /**
     * Prédit l'activité à partir d'une image.
     * 
     * @param imageData Image sous forme de tableau de pixels RGB (hauteur x largeur x 3)
     * @return Index de la classe prédite
     */
    public int predictActivity(INDArray imageData) {
        if (vgg16Network == null) {
            throw new IllegalStateException("Le modèle VGG16 n'est pas initialisé");
        }
        
        // Prétraitement de l'image pour VGG16
        VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
        preProcessor.transform(imageData);
        
        // Faire la prédiction
        INDArray output = vgg16Network.outputSingle(imageData);
        
        // Obtenir l'index de la classe avec la probabilité la plus élevée
        return output.argMax(1).getInt(0);
    }
    
    /**
     * Obtient le réseau VGG16.
     *
     * @return Le réseau VGG16
     */
    public ComputationGraph getVgg16Network() {
        return vgg16Network;
    }
}
