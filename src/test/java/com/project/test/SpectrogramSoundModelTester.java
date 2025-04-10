package com.project.test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Testeur pour le modèle de détection de sons basé sur spectrogrammes
 */
public class SpectrogramSoundModelTester extends BaseModelTester implements ModelTester {
    private static final Logger log = LoggerFactory.getLogger(SpectrogramSoundModelTester.class);
    
    private final int height;
    private final int width;
    private final int channels;
    private final String architecture;
    
    public SpectrogramSoundModelTester(Properties config) {
        super(config, "sound.spectrogram");
        
        // Charger les paramètres spécifiques au spectrogramme
        this.height = Integer.parseInt(config.getProperty("sound.spectrogram.height", "224"));
        this.width = Integer.parseInt(config.getProperty("sound.spectrogram.width", "224"));
        this.channels = 1; // Généralement en niveaux de gris
        this.architecture = config.getProperty("sound.model.architecture", "VGG16");
    }
    
    @Override
    protected double[] generateFeatureVectorForClass(int targetClass) {
        // Pour les spectrogrammes, notre implémentation générique ne fonctionnera pas directement
        // car nous avons besoin d'un tableau 3D (channels, height, width)
        // On va donc surcharger les méthodes qui utilisent ce tableau
        
        // Retourner un tableau vide
        return new double[inputSize];
    }
    
    @Override
    protected DataSet generateTestData() {
        log.info("Génération de données de test synthétiques pour les spectrogrammes");
        
        // Créer des tableaux pour les entrées et les sorties
        // Format [batchSize, channels, height, width]
        INDArray features = Nd4j.zeros(testSamples, channels, height, width);
        INDArray labels = Nd4j.zeros(testSamples, numClasses);
        
        // Générer des spectrogrammes synthétiques
        for (int i = 0; i < testSamples; i++) {
            // Déterminer la classe de cet exemple
            int targetClass = random.nextInt(numClasses);
            
            // Générer un spectrogramme qui représente cette classe
            generateSyntheticSpectrogram(features, i, targetClass);
            
            // Ajouter l'étiquette one-hot
            labels.putScalar(i, targetClass, 1.0);
        }
        
        return new DataSet(features, labels);
    }
    
    /**
     * Génère un spectrogramme synthétique pour une classe spécifique
     * 
     * @param features Tenseur d'entrée
     * @param batchIndex Indice dans le lot
     * @param targetClass Classe cible
     */
    private void generateSyntheticSpectrogram(INDArray features, int batchIndex, int targetClass) {
        // Paramètres de base pour cette classe
        double baseFrequency = 1.0 + targetClass * 2.0;
        double baseAmplitude = 0.3 + (0.6 * targetClass / (numClasses - 1));
        
        // Pour chaque pixel dans le spectrogramme
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // L'axe Y représente la fréquence (inversée: basses fréquences en bas)
                // Normaliser y pour qu'il varie de 0 à 1
                double normY = (height - 1 - y) / (double)(height - 1);
                
                // L'axe X représente le temps
                double normX = x / (double)(width - 1);
                
                // Amplitude dans cette bande de fréquence au temps donné
                double frequency = baseFrequency * (1 + normY * 3); // Plus élevé pour les hautes fréquences
                double amplitude = baseAmplitude * Math.exp(-Math.pow(normY - 0.3 * (1 + targetClass / (double)numClasses), 2) / 0.1);
                
                // Moduler l'amplitude dans le temps avec des harmoniques
                double timeModulation = Math.sin(normX * 10 * Math.PI * (1 + 0.2 * targetClass));
                
                // Valeur du pixel
                double value = amplitude * (0.7 + 0.3 * timeModulation);
                
                // Ajouter du bruit
                double noise = random.nextGaussian() * 0.05;
                
                // Normaliser entre 0 et 1
                double pixelValue = Math.max(0, Math.min(1, value + noise));
                
                // Stocker dans le tenseur
                features.putScalar(new int[]{batchIndex, 0, y, x}, pixelValue);
            }
        }
    }
    
    @Override
    protected INDArray generateRandomInput() {
        // Créer un spectrogramme aléatoire de taille [1, channels, height, width]
        INDArray input = Nd4j.rand(new int[]{1, channels, height, width});
        return input;
    }
    
    /**
     * Valide spécifiquement les aspects du modèle basé sur spectrogrammes
     */
    @Override
    public boolean validateModel() {
        if (!super.validateModel()) {
            return false;
        }
        
        // Vérifications spécifiques au modèle CNN
        try {
            log.info("Architecture du modèle: {}", architecture);
            
            // Vérifier que la forme d'entrée est correcte
            // Dans DL4J 1.0.0-beta7, nous utilisons layerInputSize() avec l'index de la couche
            long[] inputShape = model.layerInputSize(0);
            
            if (inputShape[0] != (long)channels || inputShape[1] != (long)height || inputShape[2] != (long)width) {
                log.error("La forme d'entrée du modèle ([{}, {}, {}]) ne correspond pas à la forme attendue ([{}, {}, {}]) pour les spectrogrammes", 
                        inputShape[0], inputShape[1], inputShape[2], channels, height, width);
                return false;
            }
            
            // Vérifier que c'est bien un réseau convolutif (la première couche devrait être une conv)
            String firstLayerType = model.getLayer(0).getClass().getSimpleName();
            if (!firstLayerType.toLowerCase().contains("convolution")) {
                log.warn("Le premier type de couche ({}) ne semble pas être convolutif", firstLayerType);
            }
            
            log.info("Le modèle de spectrogramme a la bonne forme d'entrée: [{}x{}x{}]", channels, height, width);
            return true;
            
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle basé sur spectrogrammes", e);
            return false;
        }
    }
}