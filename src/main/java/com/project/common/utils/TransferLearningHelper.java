package com.project.common.utils;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Classe utilitaire pour le transfert d'apprentissage avec des modèles pré-entraînés.
 * Fournit des méthodes pour charger, adapter et fine-tuner des modèles.
 */
public class TransferLearningHelper {
    private static final Logger log = LoggerFactory.getLogger(TransferLearningHelper.class);
    
    // Chemins temporaires pour stocker les modèles téléchargés
    private static final String MOBILENET_TEMP_PATH = System.getProperty("java.io.tmpdir") + "/mobilenetv2.zip";
    private static final String YAMNET_TEMP_PATH = System.getProperty("java.io.tmpdir") + "/yamnet.zip";
    
    private TransferLearningHelper() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Charge un modèle MobileNetV2 pré-entraîné et l'adapte pour la classification d'activités.
     * 
     * @param numClasses Nombre de classes dans le problème de classification
     * @param seed Graine pour l'initialisation aléatoire
     * @param learningRate Taux d'apprentissage pour les nouvelles couches
     * @return MultiLayerNetwork adapté pour la classification d'activités
     * @throws IOException en cas d'erreur lors du chargement du modèle
     */
    public static MultiLayerNetwork loadMobileNetV2ForActivityClassification(
            int numClasses, int seed, double learningRate) throws IOException {
        log.info("Chargement et adaptation du modèle MobileNetV2 pour la classification d'activités avec {} classes", numClasses);
        
        // Pour l'instant, nous utiliserons VGG16 comme exemple car MobileNetV2 n'est pas directement disponible dans DL4J Zoo
        // Dans un vrai scénario, vous devriez télécharger et importer correctement MobileNetV2
        ZooModel<?> zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        
        log.info("Modèle pré-entraîné chargé avec succès");
        
        // Configurer le transfert d'apprentissage
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(learningRate))
                .seed(seed)
                .build();
        
        // Construire un nouveau modèle en utilisant le transfert d'apprentissage
        ComputationGraph transferredModel = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2") // Extraire les caractéristiques jusqu'à cette couche
                .removeVertexAndConnections("predictions") // Supprimer la couche de sortie existante
                .addLayer("activityDense", 
                         new DenseLayer.Builder().nIn(4096).nOut(1024)
                                 .activation(Activation.RELU)
                                 .weightInit(WeightInit.XAVIER)
                                 .build(), 
                         "fc2") // Ajouter une nouvelle couche dense après fc2
                .addLayer("activityOutput", 
                         new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                 .nIn(1024).nOut(numClasses)
                                 .activation(Activation.SOFTMAX)
                                 .weightInit(WeightInit.XAVIER)
                                 .build(), 
                         "activityDense") // Ajouter une nouvelle couche de sortie
                .setOutputs("activityOutput") // Définir la sortie du réseau
                .build();
        
        log.info("Modèle adapté pour la classification d'activités");
        
        // Convertir en MultiLayerNetwork pour la compatibilité avec le reste du code
        return convertComputationGraphToMultiLayerNetwork(transferredModel);
    }
    
    /**
     * Charge un modèle YAMNet pré-entraîné et l'adapte pour la classification de sons.
     * 
     * @param numClasses Nombre de classes dans le problème de classification
     * @param seed Graine pour l'initialisation aléatoire
     * @param learningRate Taux d'apprentissage pour les nouvelles couches
     * @return MultiLayerNetwork adapté pour la classification de sons
     * @throws IOException en cas d'erreur lors du chargement du modèle
     */
    public static MultiLayerNetwork loadYAMNetForSoundClassification(
            int numClasses, int seed, double learningRate) throws IOException {
        log.info("Chargement et adaptation du modèle YAMNet pour la classification de sons avec {} classes", numClasses);
        
        // Création d'un modèle simple pour simuler YAMNet (qui n'est pas disponible directement)
        // Dans un vrai scénario, vous devriez télécharger et importer correctement YAMNet
        int inputSize = AudioUtils.YAMNET_WAVEFORM_LENGTH;
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize).nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(512).nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(256).nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.feedForward(inputSize))
                .build();
        
        MultiLayerNetwork yamnetSimulation = new MultiLayerNetwork(conf);
        yamnetSimulation.init();
        
        log.info("Modèle simulant YAMNet créé (dans un vrai scénario, il faudrait charger le vrai modèle YAMNet)");
        
        return yamnetSimulation;
    }
    
    /**
     * Convertit un ComputationGraph en MultiLayerNetwork.
     * Note: Cette conversion n'est pas toujours possible et peut perdre des fonctionnalités.
     * Elle est utilisée ici pour maintenir la compatibilité avec le reste du code.
     * 
     * @param graph ComputationGraph à convertir
     * @return MultiLayerNetwork converti
     */
    private static MultiLayerNetwork convertComputationGraphToMultiLayerNetwork(ComputationGraph graph) {
        log.info("Conversion de ComputationGraph en MultiLayerNetwork pour compatibilité");
        
        // Cette méthode est simplifiée et ne fonctionnera que pour des graphs simples
        // Dans un cas réel, vous voudrez peut-être adapter le reste du code pour utiliser ComputationGraph
        
        Layer[] layers = graph.getLayers();
        int numLayers = layers.length;
        
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(119)
                .updater(new Adam(0.001));
        
        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        
        // Ajouter chaque couche
        for (int i = 0; i < numLayers; i++) {
            // Attention: ceci est très simplifié et ne fonctionnera pas pour tous les types de couches
            // ou configurations de ComputationGraph
            listBuilder.layer(i, layers[i].conf().getLayer().clone());
        }
        
        // Définir l'entrée
        listBuilder.setInputType(graph.getConfiguration().getNetworkInputTypes()[0]);
        
        MultiLayerNetwork network = new MultiLayerNetwork(listBuilder.build());
        network.init();
        
        // Copier les paramètres du graph (ceci ne fonctionnera que pour des réseaux simples)
        for (int i = 0; i < numLayers; i++) {
            INDArray params = layers[i].params();
            if (params != null) {
                network.getLayer(i).setParams(params.dup());
            }
        }
        
        log.info("Conversion en MultiLayerNetwork terminée (attention: cette conversion est simplifiée)");
        
        return network;
    }
    
    /**
     * Vérifie si un modèle MobileNetV2 est déjà téléchargé.
     * 
     * @return true si le modèle existe, false sinon
     */
    public static boolean isMobileNetV2Downloaded() {
        File modelFile = new File(MOBILENET_TEMP_PATH);
        return modelFile.exists() && modelFile.isFile() && modelFile.length() > 0;
    }
    
    /**
     * Vérifie si un modèle YAMNet est déjà téléchargé.
     * 
     * @return true si le modèle existe, false sinon
     */
    public static boolean isYAMNetDownloaded() {
        File modelFile = new File(YAMNET_TEMP_PATH);
        return modelFile.exists() && modelFile.isFile() && modelFile.length() > 0;
    }
}
