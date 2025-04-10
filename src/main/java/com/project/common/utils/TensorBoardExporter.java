package com.project.common.utils;

import org.deeplearning4j.api.storage.StatsStorage;
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

/**
 * Classe pour exporter les métriques d'évaluation au format TensorBoard.
 * Permet de visualiser l'évolution des métriques pendant l'entraînement et l'évaluation
 * des modèles via l'interface TensorBoard.
 */
public class TensorBoardExporter {
    private static final Logger log = LoggerFactory.getLogger(TensorBoardExporter.class);
    
    private static UIServer uiServer;
    private static StatsStorage statsStorage;
    
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
            statsStorage = new FileStatsStorage(new File(logDirectory, "dl4j-stats.bin"));
            
            // Initialiser le serveur UI si ce n'est pas déjà fait
            if (uiServer == null) {
                uiServer = UIServer.getInstance();
                uiServer.attach(statsStorage);
                log.info("Serveur TensorBoard démarré - Accédez à http://localhost:9000/train pour visualiser les métriques");
            }
            
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
            statsStorage = new InMemoryStatsStorage();
            
            // Initialiser le serveur UI si ce n'est pas déjà fait
            if (uiServer == null) {
                uiServer = UIServer.getInstance();
                uiServer.attach(statsStorage);
                log.info("Serveur TensorBoard démarré en mémoire - Accédez à http://localhost:9000/train pour visualiser les métriques");
            }
            
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
        
        return new StatsListener(statsStorage, 10, modelName);
    }
    
    /**
     * Exporte une liste de métriques d'évaluation vers TensorBoard.
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
            // Note : ce n'est pas un modèle réel, juste un moyen d'envoyer des métriques à TensorBoard
            org.deeplearning4j.nn.graph.ComputationGraph dummyModel = new org.deeplearning4j.nn.graph.ComputationGraph(null);
            dummyModel.setListeners(new StatsListener(statsStorage));
            
            // Ajouter les métriques de chaque époque
            for (EvaluationMetrics metric : metrics) {
                int iteration = metric.getEpoch();
                
                // Publier les métriques globales
                String prefix = "metrics/";
                dummyModel.getListeners().get(0).iterationDone(dummyModel, iteration, iteration);
                statsStorage.putStaticInfo(
                        new org.deeplearning4j.api.storage.Persistable() {
                            @Override
                            public String getSessionID() {
                                return modelName;
                            }
                            
                            @Override
                            public String getTypeID() {
                                return "metrics";
                            }
                            
                            @Override
                            public String getWorkerID() {
                                return "global";
                            }
                            
                            @Override
                            public long getTimeStamp() {
                                return System.currentTimeMillis();
                            }
                            
                            @Override
                            public void setTimeStamp(long timeStamp) {
                                // Non utilisé
                            }
                            
                            @Override
                            public void setSessionID(String sessionID) {
                                // Non utilisé
                            }
                            
                            @Override
                            public void setTypeID(String typeID) {
                                // Non utilisé
                            }
                            
                            @Override
                            public void setWorkerID(String workerID) {
                                // Non utilisé
                            }
                            
                            @Override
                            public byte[] encode() {
                                return new byte[0]; // Non utilisé
                            }
                            
                            @Override
                            public void decode(byte[] encoded) {
                                // Non utilisé
                            }
                        }
                );
                
                // Publier les métriques sous forme de scalaires
                statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                        prefix + "accuracy", metric.getAccuracy());
                statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                        prefix + "precision", metric.getPrecision());
                statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                        prefix + "recall", metric.getRecall());
                statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                        prefix + "f1_score", metric.getF1Score());
                statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                        prefix + "training_time", metric.getTrainingTime());
                
                // Ajouter les métriques par classe si disponibles
                for (int classIdx : metric.getPerClassMetrics().keySet()) {
                    EvaluationMetrics.ClassMetrics classMetrics = metric.getClassMetrics(classIdx);
                    String classPrefix = prefix + "class_" + classIdx + "/";
                    
                    statsStorage.putScalar(modelName, "metrics", "class_" + classIdx, iteration, 
                            classPrefix + "precision", classMetrics.getPrecision());
                    statsStorage.putScalar(modelName, "metrics", "class_" + classIdx, iteration, 
                            classPrefix + "recall", classMetrics.getRecall());
                    statsStorage.putScalar(modelName, "metrics", "class_" + classIdx, iteration, 
                            classPrefix + "f1_score", classMetrics.getF1Score());
                }
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
            String prefix = "metrics/";
            
            // Publier les métriques globales
            statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                    prefix + "accuracy", evaluation.accuracy());
            statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                    prefix + "precision", evaluation.precision());
            statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                    prefix + "recall", evaluation.recall());
            statsStorage.putScalar(modelName, "metrics", "global", iteration, 
                    prefix + "f1_score", evaluation.f1());
            
            // Récupérer le nombre de classes
            int numClasses = evaluation.getNumRowCounter().rows();
            
            // Ajouter les métriques par classe
            for (int i = 0; i < numClasses; i++) {
                String classPrefix = prefix + "class_" + i + "/";
                
                statsStorage.putScalar(modelName, "metrics", "class_" + i, iteration, 
                        classPrefix + "precision", evaluation.precision(i));
                statsStorage.putScalar(modelName, "metrics", "class_" + i, iteration, 
                        classPrefix + "recall", evaluation.recall(i));
                statsStorage.putScalar(modelName, "metrics", "class_" + i, iteration, 
                        classPrefix + "f1_score", evaluation.f1(i));
            }
            
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
