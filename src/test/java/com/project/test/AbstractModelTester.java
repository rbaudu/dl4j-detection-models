package com.project.test;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Classe abstraite implémentant les méthodes communes pour tester les modèles
 */
public abstract class AbstractModelTester implements ModelTester {
    
    protected static final Logger log = LoggerFactory.getLogger(AbstractModelTester.class);
    
    protected final Properties config;
    protected MultiLayerNetwork model;
    protected String modelPath;
    protected String testDataPath;
    
    public AbstractModelTester(Properties config) {
        this.config = config;
    }
    
    @Override
    public boolean loadModel() {
        try {
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                log.error("Le fichier du modèle n'existe pas: {}", modelPath);
                return false;
            }
            
            model = MultiLayerNetwork.load(modelFile, true);
            log.info("Modèle chargé avec succès: {}", modelPath);
            return true;
        } catch (IOException e) {
            log.error("Erreur lors du chargement du modèle", e);
            return false;
        }
    }
    
    @Override
    public EvaluationResult testModel() {
        if (model == null) {
            if (!loadModel()) {
                log.error("Impossible de charger le modèle pour le test");
                return null;
            }
        }
        
        EvaluationResult result = new EvaluationResult();
        
        try {
            // Créer un DataSetIterator à partir des données de test
            DataSetIterator testIterator = createTestDataIterator();
            
            // Évaluer le modèle
            Evaluation eval = model.evaluate(testIterator);
            
            // Remplir les résultats
            result.setAccuracy(eval.accuracy());
            result.setPrecision(eval.precision());
            result.setRecall(eval.recall());
            result.setF1Score(eval.f1());
            
            // Ajouter les précisions par classe
            int numClasses = eval.numClasses();
            for (int i = 0; i < numClasses; i++) {
                String label = String.valueOf(i); // Utiliser l'index comme nom de classe
                
                // Calculer la précision par classe avec des méthodes alternatives
                double truePositives = eval.truePositives().get(i);
                double falseNegatives = eval.falseNegatives().get(i);
                double classAccuracy = truePositives / (truePositives + falseNegatives);
                
                result.addClassAccuracy(label, classAccuracy);
            }
            
            // Ajouter des métriques supplémentaires
            result.addMetric("Matrix de confusion", eval.getConfusionMatrix().toCSV());
            
            log.info("Test du modèle terminé avec précision: {}%", result.getAccuracy() * 100);
            
        } catch (Exception e) {
            log.error("Erreur lors du test du modèle", e);
        }
        
        return result;
    }
    
    @Override
    public boolean validateModel() {
        if (model == null) {
            if (!loadModel()) {
                return false;
            }
        }
        
        try {
            // Vérifier si le modèle peut faire une prédiction simple
            DataSetIterator testIterator = createTestDataIterator();
            if (testIterator.hasNext()) {
                model.output(testIterator.next().getFeatures());
                log.info("Le modèle peut effectuer des prédictions");
                return true;
            } else {
                log.error("Aucune donnée de test disponible pour la validation");
                return false;
            }
        } catch (Exception e) {
            log.error("Erreur lors de la validation du modèle", e);
            return false;
        }
    }
    
    /**
     * Crée un DataSetIterator pour les données de test
     * 
     * @return DataSetIterator pour les données de test
     */
    protected abstract DataSetIterator createTestDataIterator() throws IOException;
}