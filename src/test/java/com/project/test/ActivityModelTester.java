package com.project.test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Testeur pour le modèle de détection d'activité
 */
public class ActivityModelTester extends ModelTester {
    private static final Logger log = LoggerFactory.getLogger(ActivityModelTester.class);
    
    public ActivityModelTester(Properties config) {
        super(config, "activity");
    }
    
    @Override
    protected double[] generateFeatureVectorForClass(int targetClass) {
        double[] features = new double[inputSize];
        
        // Pour l'activité, on peut simuler différents motifs
        for (int i = 0; i < inputSize; i++) {
            // Valeur de base pour chaque classe
            double baseValue = 0.2 + (targetClass * 0.6 / (numClasses - 1));
            
            // Ajouter un peu de variabilité selon la position dans le vecteur
            double positionEffect = Math.sin(i * 2 * Math.PI / inputSize) * 0.2;
            
            // Ajouter du bruit
            double noise = random.nextGaussian() * 0.1;
            
            // Combiner les effets
            features[i] = Math.max(0, Math.min(1, baseValue + positionEffect + noise));
        }
        
        return features;
    }
}