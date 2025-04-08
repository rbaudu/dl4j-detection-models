package com.project.training;

import com.project.common.utils.DataProcessor;
import com.project.common.utils.ModelUtils;
import com.project.models.presence.PresenceModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.Random;

/**
 * Classe pour l'entraînement du modèle de détection de présence.
 * Implémente les méthodes spécifiques pour préparer les données et gérer le modèle de présence.
 */
public class PresenceTrainer extends ModelTrainer {
    private static final Logger log = LoggerFactory.getLogger(PresenceTrainer.class);
    
    private final PresenceModel model;
    private final String dataDir;
    private final String modelDir;
    private final String modelName;
    
    /**
     * Constructeur avec configuration.
     *
     * @param config Propriétés de configuration
     */
    public PresenceTrainer(Properties config) {
        super(config);
        this.model = new PresenceModel(config);
        this.dataDir = config.getProperty("presence.data.dir", "data/raw/presence");
        this.modelDir = config.getProperty("presence.model.dir", "models/presence");
        this.modelName = config.getProperty("presence.model.name", "presence_model");
        
        try {
            createDirectories(modelDir);
        } catch (IOException e) {
            log.error("Erreur lors de la création des répertoires", e);
        }
    }
    
    @Override
    protected DataSet prepareData() throws IOException {
        log.info("Préparation des données pour le modèle de détection de présence");
        
        // Charger les fichiers de données
        List<File> dataFiles = DataProcessor.loadDataFiles(dataDir);
        
        // Vérifier si des fichiers de données ont été trouvés
        if (dataFiles.isEmpty()) {
            log.error("Aucun fichier de données trouvé dans {}", dataDir);
            return null;
        }
        
        // Traiter les fichiers de données
        int inputSize = Integer.parseInt(config.getProperty("presence.model.input.size", "64"));
        int totalExamples = 1000; // Nombre d'exemples à générer (à ajuster selon les besoins)
        
        // Pour cet exemple, nous allons générer des données synthétiques
        // Dans un cas réel, vous chargeriez et prétraiteriez vos vraies données ici
        return generateSyntheticData(inputSize, totalExamples);
    }
    
    /**
     * Génère des données synthétiques pour l'entraînement.
     * Cette méthode est utilisée uniquement pour démontrer le flux de travail.
     * Dans un cas réel, vous utiliseriez vos propres données.
     *
     * @param inputSize Taille de l'entrée du modèle
     * @param numExamples Nombre d'exemples à générer
     * @return DataSet contenant les données synthétiques
     */
    private DataSet generateSyntheticData(int inputSize, int numExamples) {
        log.info("Génération de données synthétiques avec {} exemples", numExamples);
        
        // Créer des tableaux pour les entrées et les sorties
        INDArray features = Nd4j.zeros(numExamples, inputSize);
        INDArray labels = Nd4j.zeros(numExamples, 2); // 2 classes: absence/présence
        
        Random random = new Random(42); // Graine fixe pour la reproductibilité
        
        // Générer des caractéristiques et des étiquettes aléatoires
        for (int i = 0; i < numExamples; i++) {
            // Décider si cet exemple représente une présence (1) ou une absence (0)
            boolean isPresent = random.nextDouble() > 0.5;
            
            // Générer des caractéristiques bruitées en fonction de la classe
            double baseValue = isPresent ? 0.7 : 0.3;
            
            for (int j = 0; j < inputSize; j++) {
                // Ajouter du bruit à la valeur de base
                double noise = random.nextGaussian() * 0.1;
                double value = Math.max(0, Math.min(1, baseValue + noise));
                features.putScalar(i, j, value);
            }
            
            // Définir l'étiquette one-hot
            labels.putScalar(i, isPresent ? 1 : 0, 1.0);
        }
        
        return new DataSet(features, labels);
    }
    
    @Override
    protected MultiLayerNetwork getModel() {
        // Initialiser un nouveau modèle
        model.initNewModel();
        return model.getNetwork();
    }
    
    @Override
    protected void saveModel(MultiLayerNetwork network) throws IOException {
        String modelPath = new File(modelDir, modelName + ".zip").getPath();
        log.info("Sauvegarde du modèle final vers {}", modelPath);
        ModelUtils.saveModel(network, modelPath);
    }
    
    @Override
    protected void saveCheckpoint(MultiLayerNetwork network, int epoch) throws IOException {
        String checkpointPath = new File(modelDir + "/checkpoints", 
                                          modelName + "_epoch_" + epoch + ".zip").getPath();
        log.info("Sauvegarde du checkpoint à l'époque {} vers {}", epoch, checkpointPath);
        ModelUtils.saveModel(network, checkpointPath);
    }
}
