package com.project.test;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Testeur pour le modèle de détection de sons
 */
public class SoundModelTester extends BaseModelTester implements ModelTester {
    private static final Logger log = LoggerFactory.getLogger(SoundModelTester.class);
    
    public SoundModelTester(Properties config) {
        super(config, "sound");
    }
    
    @Override
    protected double[] generateFeatureVectorForClass(int targetClass) {
        double[] features = new double[inputSize];
        
        // Pour les sons, simuler différentes fréquences dominantes
        for (int i = 0; i < inputSize; i++) {
            // Amplitude de base pour cette classe
            double baseAmplitude = 0.3 + (0.6 * targetClass / (numClasses - 1));
            
            // Fréquence "caractéristique" pour cette classe
            double frequency = 1.0 + targetClass * 2.0;
            
            // Générer une forme d'onde avec cette fréquence
            double wave = Math.sin(i * frequency * 2 * Math.PI / inputSize) * baseAmplitude;
            
            // Ajouter du bruit
            double noise = random.nextGaussian() * 0.1;
            
            // Limiter les valeurs entre 0 et 1
            features[i] = Math.max(0, Math.min(1, 0.5 + wave + noise));
        }
        
        return features;
    }
    
/*    @Override
    protected DataSetIterator createTestDataIterator() throws IOException {
        // Utiliser la méthode de génération de données synthétiques de BaseModelTester
        return new org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator(generateTestData(), batchSize);
    }*/
}