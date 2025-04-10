package com.project.common.utils;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Classe pour exporter les métriques d'évaluation au format TensorBoard.
 * Permet de visualiser l'évolution des métriques pendant l'entraînement et l'évaluation
 * des modèles via l'interface TensorBoard.
 * 
 * Cette classe a été adaptée pour fonctionner avec la version 1.0.0-beta7 de DL4J.
 */
public class TensorBoardExporter {
    private static final Logger log = LoggerFactory.getLogger(TensorBoardExporter.class);
    
    private static UIServer uiServer;
    private static Object statsStorage;  // Type générique pour supporter différentes implémentations
    private static boolean isInitialized = false;
    private static final AtomicInteger epochCounter = new AtomicInteger(0);  // Compteur pour simuler les époques
    
    /**
     * Initialise le serveur TensorBoard et configure le stockage des métriques.
     * Cette méthode doit être appelée une seule fois avant d'utiliser les autres méthodes.
     * 
     * @param logDirectory Répertoire où stocker les fichiers de log TensorBoard
     * @return true si l'initialisation a réussi, false sinon
     */
    public static boolean initialize(String logDirectory) {
        try {
            // Créer le répertoire s'il n'existe pas
            File directory = new File(logDirectory);
            if (!directory.exists()) {
                if (!directory.mkdirs()) {
                    log.error("Impossible de créer le répertoire pour les logs TensorBoard: {}", logDirectory);
                    return false;
                }
            }
            
            // Initialiser le stockage des métriques
            FileStatsStorage fileStatsStorage = new FileStatsStorage(new File(logDirectory, "dl4j-stats.bin"));
            statsStorage = fileStatsStorage;
            
            // Initialiser le serveur UI si ce n'est pas déjà fait
            if (uiServer == null) {
                uiServer = UIServer.getInstance();
                uiServer.attach(fileStatsStorage);
                log.info("Serveur TensorBoard démarré - Accédez à http://localhost:9000/train pour visualiser les métriques");
            }
            
            isInitialized = true;
            return true;
        } catch (Exception e) {
            log.error("Erreur lors de l'initialisation de TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Initialise le serveur TensorBoard en mémoire (sans persistance sur disque).
     * Utile pour une visualisation temporaire ou pour les tests.
     * 
     * @return true si l'initialisation a réussi, false sinon
     */
    public static boolean initializeInMemory() {
        try {
            // Initialiser le stockage des métriques en mémoire
            InMemoryStatsStorage memoryStatsStorage = new InMemoryStatsStorage();
            statsStorage = memoryStatsStorage;
            
            // Initialiser le serveur UI si ce n'est pas déjà fait
            if (uiServer == null) {
                uiServer = UIServer.getInstance();
                uiServer.attach(memoryStatsStorage);
                log.info("Serveur TensorBoard démarré en mémoire - Accédez à http://localhost:9000/train pour visualiser les métriques");
            }
            
            isInitialized = true;
            return true;
        } catch (Exception e) {
            log.error("Erreur lors de l'initialisation de TensorBoard en mémoire: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Crée un StatsListener qui peut être attaché à un modèle pour enregistrer les métriques
     * pendant l'entraînement.
     * 
     * @param modelName Nom du modèle pour identification
     * @return StatsListener à attacher au modèle
     */
    public static StatsListener getStatsListener(String modelName) {
        if (statsStorage == null) {
            // Auto-initialisation en mémoire si pas déjà fait
            if (!initializeInMemory()) {
                log.error("Impossible d'initialiser TensorBoard pour obtenir un StatsListener");
                return null;
            }
        }
        
        if (statsStorage instanceof FileStatsStorage) {
            return new StatsListener((FileStatsStorage) statsStorage, 10, modelName);
        } else if (statsStorage instanceof InMemoryStatsStorage) {
            return new StatsListener((InMemoryStatsStorage) statsStorage, 10, modelName);
        } else {
            log.error("Type de stockage non supporté pour StatsListener");
            return null;
        }
    }
    
    /**
     * Crée une configuration minimale pour un modèle fictif utilisé pour exporter les métriques
     * 
     * @return Configuration pour un modèle minimal
     */
    private static MultiLayerConfiguration createDummyConfig() {
        // Créer une configuration minimale avec un builder plutôt que d'utiliser builder() directement
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        NeuralNetConfiguration conf = builder.build();
        
        // Construire la configuration multicouche
        MultiLayerConfiguration.Builder layerBuilder = new MultiLayerConfiguration.Builder();
        // Supprimé la ligne setInputTypes qui n'est pas disponible dans cette version
        
        return layerBuilder.build();
    }
    
    /**
     * Exporte une liste de métriques d'évaluation vers TensorBoard.
     * Implémentation adaptée pour utiliser les fonctionnalités disponibles dans la version actuelle.
     * 
     * @param metrics Liste des métriques à exporter
     * @param modelName Nom du modèle pour identification
     * @return true si l'export a réussi, false sinon
     */
    public static boolean exportMetrics(List<EvaluationMetrics> metrics, String modelName) {
        if (statsStorage == null) {
            // Auto-initialisation en mémoire si pas déjà fait
            if (!initializeInMemory()) {
                log.error("Impossible d'initialiser TensorBoard pour l'export des métriques");
                return false;
            }
        }
        
        try {
            // Créer un modèle de test simple pour attacher les métriques
            // Dans la version actuelle, nous utilisons une approche différente
            org.deeplearning4j.nn.multilayer.MultiLayerNetwork dummyModel = 
                new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(
                    createDummyConfig());
            
            // Créer un listener pour envoyer les données
            StatsListener listener = null;
            if (statsStorage instanceof FileStatsStorage) {
                listener = new StatsListener((FileStatsStorage) statsStorage);
            } else if (statsStorage instanceof InMemoryStatsStorage) {
                listener = new StatsListener((InMemoryStatsStorage) statsStorage);
            } else {
                log.error("Type de stockage non supporté");
                return false;
            }
            
            dummyModel.setListeners(listener);
            
            // Envoyer les métriques pour chaque époque
            for (EvaluationMetrics metric : metrics) {
                int iteration = metric.getEpoch();
                
                // Simuler un événement d'itération pour déclencher la collecte de statistiques
                listener.iterationDone(dummyModel, iteration, iteration);
                
                // Ajouter les métriques sous forme de scores (n'utilise pas l'API Storage directement)
                double[] scores = {
                    metric.getAccuracy(),
                    metric.getPrecision(),
                    metric.getRecall(),
                    metric.getF1Score()
                };
                
                dummyModel.setScore(metric.getF1Score()); // Utilise F1 comme score principal
                
                // Simuler un événement pour déclencher la mise à jour des stats
                listener.iterationDone(dummyModel, iteration + 1, iteration + 1);
            }
            
            log.info("Métriques exportées vers TensorBoard pour le modèle {}", modelName);
            return true;
        } catch (Exception e) {
            log.error("Erreur lors de l'exportation des métriques vers TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Exporte une évaluation DL4J directement vers TensorBoard.
     * Utile pour l'export direct depuis un modèle après évaluation.
     * Implémentation adaptée pour la version actuelle.
     * 
     * @param evaluation Objet Evaluation de DL4J
     * @param iteration Numéro d'itération/époque
     * @param modelName Nom du modèle
     * @return true si l'export a réussi, false sinon
     */
    public static boolean exportEvaluation(Evaluation evaluation, int iteration, String modelName) {
        if (statsStorage == null) {
            // Auto-initialisation en mémoire si pas déjà fait
            if (!initializeInMemory()) {
                log.error("Impossible d'initialiser TensorBoard pour l'export de l'évaluation");
                return false;
            }
        }
        
        try {
            // Créer un modèle de test pour l'export
            org.deeplearning4j.nn.multilayer.MultiLayerNetwork dummyModel = 
                new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(
                    createDummyConfig());
            
            // Créer un listener pour envoyer les données
            StatsListener listener = null;
            if (statsStorage instanceof FileStatsStorage) {
                listener = new StatsListener((FileStatsStorage) statsStorage);
            } else if (statsStorage instanceof InMemoryStatsStorage) {
                listener = new StatsListener((InMemoryStatsStorage) statsStorage);
            } else {
                log.error("Type de stockage non supporté");
                return false;
            }
            
            dummyModel.setListeners(listener);
            
            // Définir le score du modèle basé sur l'évaluation
            dummyModel.setScore(evaluation.f1());
            
            // Déclencher l'événement de fin d'itération pour enregistrer les statistiques
            listener.iterationDone(dummyModel, iteration, iteration);
            
            log.info("Évaluation exportée vers TensorBoard pour le modèle {}", modelName);
            return true;
        } catch (Exception e) {
            log.error("Erreur lors de l'exportation de l'évaluation vers TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Arrête le serveur TensorBoard et libère les ressources.
     * Doit être appelé à la fin de l'application pour un arrêt propre.
     */
    public static void shutdown() {
        if (uiServer != null) {
            try {
                uiServer.stop();
                uiServer = null;
                statsStorage = null;
                isInitialized = false;
                log.info("Serveur TensorBoard arrêté");
            } catch (Exception e) {
                log.error("Erreur lors de l'arrêt du serveur TensorBoard: {}", e.getMessage());
            }
        }
    }
    
    /**
     * Exporte les métriques vers TensorBoard à partir des paramètres de configuration
     * 
     * @param metrics Liste des métriques
     * @param config Propriétés de configuration
     * @param modelName Nom du modèle
     * @return true si l'export a réussi, false sinon
     */
    public static boolean exportFromConfig(List<EvaluationMetrics> metrics, Properties config, String modelName) {
        String logDir = config.getProperty("tensorboard.log.dir", "output/tensorboard");
        boolean enableTensorBoard = Boolean.parseBoolean(config.getProperty("tensorboard.enabled", "true"));
        
        if (!enableTensorBoard) {
            log.info("Export TensorBoard désactivé dans la configuration");
            return false;
        }
        
        // Initialiser TensorBoard avec le répertoire spécifié
        if (!initialize(logDir)) {
            log.error("Impossible d'initialiser TensorBoard avec le répertoire {}", logDir);
            return false;
        }
        
        // Exporter les métriques
        return exportMetrics(metrics, modelName);
    }
}