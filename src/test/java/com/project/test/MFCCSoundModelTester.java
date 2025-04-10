package com.project.test;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Properties;

/**
 * Testeur pour le modèle de détection de sons basé sur MFCC
 */
public class MFCCSoundModelTester extends BaseModelTester implements ModelTester {
    private static final Logger log = LoggerFactory.getLogger(MFCCSoundModelTester.class);
    
    private final int numMfcc;
    private final int inputLength;
    
    public MFCCSoundModelTester(Properties config) {
        super(config, "sound.mfcc");
        // Charger les paramètres spécifiques à MFCC
        this.numMfcc = Integer.parseInt(config.getProperty("sound.num.mfcc", "40"));
        this.inputLength = Integer.parseInt(config.getProperty("sound.input.length", "16000"));
    }
    
    @Override
    protected double[] generateFeatureVectorForClass(int targetClass) {
        int mfccInputSize = numMfcc * inputLength;
        double[] features = new double[mfccInputSize];
        
        // Pour les MFCC, simuler différentes caractéristiques fréquentielles
        for (int i = 0; i < mfccInputSize; i++) {
            // Calculer l'indice MFCC et l'indice temporel
            int mfccIndex = i % numMfcc;
            int timeIndex = i / numMfcc;
            
            // Amplitude de base pour cette classe
            double baseAmplitude = 0.3 + (0.6 * targetClass / (numClasses - 1));
            
            // Les coefficients MFCC varient selon la classe
            double classModulation = (targetClass + 1) / (double) numClasses;
            
            // Moduler les MFCC en fonction de la classe
            double value = 0;
            
            // Les coefficients de bas ordre capturent l'enveloppe spectrale générale
            if (mfccIndex < 5) {
                value = Math.sin(mfccIndex * classModulation * Math.PI) * baseAmplitude;
            } 
            // Les coefficients d'ordre moyen représentent les détails de formants
            else if (mfccIndex < 20) {
                value = Math.cos(mfccIndex * (classModulation + 0.5) * Math.PI / 10) * baseAmplitude * 0.7;
            }
            // Les coefficients d'ordre élevé représentent des détails fins
            else {
                value = Math.sin(mfccIndex * (classModulation + 1) * Math.PI / 20) * baseAmplitude * 0.5;
            }
            
            // Ajouter une variation temporelle
            value *= Math.sin(timeIndex * Math.PI / inputLength) + 0.5;
            
            // Ajouter du bruit
            double noise = random.nextGaussian() * 0.05;
            
            // Normaliser entre 0 et 1
            features[i] = Math.max(0, Math.min(1, 0.5 + value + noise));
        }
        
        return features;
    }
    
    /**
     * Valide spécifiquement les aspects du modèle MFCC
     */
    @Override
    public boolean validateModel() {
        if (!super.validateModel()) {
            return false;
        }
        
        // Vérifications spécifiques au modèle MFCC
        try {
            // Vérifier que la forme d'entrée est correcte
            long[] inputShape = model.layerInputSize(0);
            long expectedInputSize = numMfcc * inputLength;
            
            if (inputShape[0] != expectedInputSize) {
                log.error("La taille d'entrée du modèle ({}) ne correspond pas à la taille attendue ({}) pour les MFCC", 
                        inputShape[0], expectedInputSize);
                return false;
            }
            
            log.info("Le modèle MFCC a la bonne forme d'entrée : {}", inputShape[0]);
            return true;
            
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle MFCC", e);
            return false;
        }
    }
}