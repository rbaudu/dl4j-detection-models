package com.project.test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Testeur pour le modèle de détection de présence
 */
public class PresenceModelTester extends ModelTester {
    private static final Logger log = LoggerFactory.getLogger(PresenceModelTester.class);
    
    public PresenceModelTester(Properties config) {
        super(config, "presence");
    }
    
    @Override
    protected double[] generateFeatureVectorForClass(int targetClass) {
        double[] features = new double[inputSize];
        
        // Valeur de base différente selon la classe
        double baseValue = (targetClass == 1) ? 0.7 : 0.3;
        
        // Générer les caractéristiques
        for (int i = 0; i < inputSize; i++) {
            // Ajouter du bruit à la valeur de base
            double noise = random.nextGaussian() * 0.1;
            features[i] = Math.max(0, Math.min(1, baseValue + noise));
        }
        
        return features;
    }
}