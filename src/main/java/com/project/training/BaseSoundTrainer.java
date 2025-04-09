package com.project.training;

import com.project.common.utils.AudioUtils;
import com.project.common.utils.DataLoaderUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;

/**
 * Classe de base abstraite pour l'entraînement des modèles de reconnaissance sonore.
 * Cette classe définit la structure commune à tous les types de traitements audio.
 */
public abstract class BaseSoundTrainer extends ModelTrainer {
    protected static final Logger log = LoggerFactory.getLogger(BaseSoundTrainer.class);
    
    // Paramètres communs à tous les types de traitements audio
    protected int numClasses;
    protected boolean useSpectrogramModel;
    protected Map<Integer, String> labelMap;
    protected Map<String, Integer> classIndices;
    
    /**
     * Constructeur avec paramètres individuels
     * 
     * @param batchSize Taille du lot pour l'entraînement
     * @param numEpochs Nombre d'époques d'entraînement
     * @param modelOutputPath Chemin où sauvegarder le modèle
     * @param numClasses Nombre de classes de sortie
     * @param useSpectrogramModel Utiliser un modèle de spectrogramme (sinon MFCC)
     */
    public BaseSoundTrainer(int batchSize, int numEpochs, String modelOutputPath, int numClasses, boolean useSpectrogramModel) {
        super(batchSize, numEpochs, modelOutputPath);
        this.numClasses = numClasses;
        this.useSpectrogramModel = useSpectrogramModel;
    }
    
    /**
     * Constructeur avec configuration
     * 
     * @param config Propriétés de configuration
     */
    public BaseSoundTrainer(Properties config) {
        super(config);
        this.numClasses = Integer.parseInt(config.getProperty("sound.model.num.classes", "8"));
        
        // Déterminer si on utilise un modèle de spectrogramme
        String modelType = config.getProperty("sound.model.type", "STANDARD");
        this.useSpectrogramModel = "SPECTROGRAM".equalsIgnoreCase(modelType);
    }
    
    /**
     * Entraîne le modèle sur les données audio
     * 
     * @param dataDir Répertoire contenant les fichiers audio
     * @param trainTestRatio Ratio pour la division train/test
     * @throws IOException Si une erreur survient lors de la lecture des données ou de la sauvegarde du modèle
     */
    public void trainOnSoundData(String dataDir, double trainTestRatio) throws IOException {
        log.info("Démarrage de l'entraînement audio sur le répertoire: {}", dataDir);
        
        // Vérifier que le répertoire existe
        File dataDirFile = new File(dataDir);
        if (!dataDirFile.exists()) {
            throw new IOException("Le répertoire de données n'existe pas: " + dataDir);
        }
        
        // Extraire les classes d'activités à partir des sous-répertoires
        classIndices = DataLoaderUtils.extractLabelsFromDirectories(dataDir);
        if (classIndices.isEmpty()) {
            throw new IOException("Aucune classe d'activité trouvée dans le répertoire: " + dataDir);
        }
        
        // Mettre à jour le nombre de classes si nécessaire
        numClasses = classIndices.size();
        log.info("Nombre de classes détectées: {}", numClasses);
        
        // Créer la carte inverse (indice -> nom de classe)
        labelMap = AudioUtils.createLabelMap(dataDirFile);
        
        // Préparer les données audio
        DataSet[] data = prepareAudioData(dataDir, dataDirFile.getAbsolutePath(), trainTestRatio);
        if (data == null || data[0].numExamples() == 0) {
            throw new IOException("Échec de la préparation des données audio");
        }
        
        DataSet trainData = data[0];
        DataSet testData = data[1];
        
        log.info("Données préparées: {} exemples d'entraînement, {} exemples de test",
                trainData.numExamples(), testData.numExamples());
        
        // Initialiser le modèle
        initializeModel();
        
        // Entraîner le modèle
        train(trainData, testData);
    }
    
    /**
     * Initialise le modèle approprié selon le type (MFCC ou Spectrogramme)
     * Cette méthode détermine quel type de modèle initialiser en fonction de la configuration
     */
    public abstract void initializeModel();
    
    /**
     * Prépare les données audio pour l'entraînement
     * 
     * @param sourceDir Répertoire source contenant les fichiers audio
     * @param basePath Chemin de base pour les fichiers audio
     * @param trainTestRatio Ratio pour la division train/test
     * @return Tableau contenant les données d'entraînement et de test
     * @throws IOException Si une erreur survient lors de la préparation des données
     */
    protected abstract DataSet[] prepareAudioData(String sourceDir, String basePath, double trainTestRatio) throws IOException;
    
    @Override
    protected DataSet prepareData() throws IOException {
        // Utiliser les répertoires spécifiés dans la configuration
        String sourceDir = config.getProperty("sound.data.dir", "data/raw/sound");
        
        try {
            // Préparer les données
            DataSet[] datasets = prepareAudioData(sourceDir, new File(sourceDir).getAbsolutePath(), 0.8);
            
            // Créer la carte des labels
            classIndices = DataLoaderUtils.extractLabelsFromDirectories(sourceDir);
            labelMap = AudioUtils.createLabelMap(new File(sourceDir));
            
            // Retourner l'ensemble complet en fusionnant train et test
            return DataSet.merge(java.util.Arrays.asList(datasets));
        } catch (Exception e) {
            log.error("Erreur lors de la préparation des données audio", e);
            throw new IOException("Erreur lors de la préparation des données audio", e);
        }
    }
}
