package com.project.common.utils;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.StatsStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Exporteur TensorBoard pour DL4J
 */
public class TensorBoardExporter {
    private static final Logger logger = LoggerFactory.getLogger(TensorBoardExporter.class);
    
    private static final String DEFAULT_TB_DIR = "tensorboard";
    private static TensorBoardExporter instance;
    private String tensorboardDir;
    private boolean inMemory;
    private static AtomicBoolean initialized = new AtomicBoolean(false);
    private static UIServer uiServer;
    private static FileStatsStorage statsStorage;
    private static StatsStorage inMemoryStatsStorage;
    
    /**
     * Constructeur par défaut
     */
    public TensorBoardExporter() {
        this(null, true);
    }
    
    /**
     * Constructeur avec répertoire personnalisé
     */
    public TensorBoardExporter(String tensorboardDir) {
        this(tensorboardDir, false);
    }
    
    /**
     * Constructeur complet
     */
    public TensorBoardExporter(String tensorboardDir, boolean inMemory) {
        this.tensorboardDir = tensorboardDir != null ? tensorboardDir : DEFAULT_TB_DIR;
        this.inMemory = inMemory;
    }
    
    /**
     * Initialise le serveur (version simplifiée)
     */
    public boolean initialize() {
        if (initialized.get()) {
            logger.info("TensorBoard déjà initialisé");
            return true;
        }
        
        try {
            // Créer le répertoire si nécessaire
            if (!inMemory) {
                File tbDir = new File(tensorboardDir);
                if (!tbDir.exists()) {
                    tbDir.mkdirs();
                }
                
                // Initialiser le stockage de statistiques
                statsStorage = new FileStatsStorage(new File(tbDir, "ui-stats.bin"));
                
                // Obtenir une instance du serveur UI
                uiServer = UIServer.getInstance();
                
                // Attacher le stockage au serveur
                uiServer.attach(statsStorage);
                
                logger.info("Serveur TensorBoard démarré - Accédez à http://localhost:9000/train pour visualiser les métriques");
            } else {
                // Stockage en mémoire
                inMemoryStatsStorage = new InMemoryStatsStorage();
                
                // Obtenir une instance du serveur UI
                uiServer = UIServer.getInstance();
                
                // Attacher le stockage au serveur
                uiServer.attach(inMemoryStatsStorage);
                
                logger.info("Serveur TensorBoard démarré en mémoire - Accédez à http://localhost:9000/train pour visualiser les métriques");
            }
            
            initialized.set(true);
            return true;
        } catch (Exception e) {
            logger.error("Erreur lors de l'initialisation de TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Arrête le serveur
     */
    public void shutdown() {
        if (initialized.get()) {
            try {
                if (uiServer != null) {
                    uiServer.stop();
                }
                logger.info("Serveur TensorBoard arrêté");
                initialized.set(false);
            } catch (Exception e) {
                logger.error("Erreur lors de l'arrêt de TensorBoard: {}", e.getMessage());
            }
        }
    }
    
    /**
     * Exporte des métriques d'évaluation
     */
    public boolean exportEvaluation(Evaluation eval, int epoch) {
        if (!initialized.get() && !initialize()) {
            return false;
        }
        
        try {
            if (eval == null) {
                throw new IllegalArgumentException("Évaluation invalide ou vide");
            }
            
            double accuracy = eval.accuracy();
            double precision = eval.precision();
            double recall = eval.recall();
            double f1 = eval.f1();
            
            logger.info("Époque {} - Accuracy: {}, Precision: {}, Recall: {}, F1: {}", 
                       epoch, accuracy, precision, recall, f1);
            
            return true;
        } catch (Exception e) {
            logger.error("Erreur lors de l'exportation de l'évaluation vers TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Exporte des métriques spécifiques
     */
    public boolean exportMetrics(Map<String, Float> metrics, int iteration) {
        if (!initialized.get() && !initialize()) {
            return false;
        }
        
        try {
            if (metrics == null || metrics.isEmpty()) {
                throw new IllegalArgumentException("Métriques invalides ou vides");
            }
            
            for (Map.Entry<String, Float> entry : metrics.entrySet()) {
                logger.info("Itération {} - {}: {}", iteration, entry.getKey(), entry.getValue());
            }
            
            return true;
        } catch (Exception e) {
            logger.error("Erreur lors de l'exportation des métriques vers TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Crée un listener pour attacher à un modèle
     */
    public StatsListener createListener(String modelName) {
        if (!initialized.get() && !initialize()) {
            logger.warn("TensorBoard n'est pas initialisé, impossible de créer un listener");
            return null;
        }
        
        if (inMemory) {
            // Dans le cas d'un stockage en mémoire
            return new StatsListener(inMemoryStatsStorage);
        } else {
            // Dans le cas d'un stockage fichier
            return new StatsListener(statsStorage);
        }
    }
    
    /**
     * Initialise à partir d'une configuration
     */
    public static TensorBoardExporter fromConfig(Properties config) {
        String tbDir = config.getProperty("tensorboard.dir", DEFAULT_TB_DIR);
        boolean inMemory = Boolean.parseBoolean(config.getProperty("tensorboard.in.memory", "false"));
        
        return new TensorBoardExporter(tbDir, inMemory);
    }
    
    /**
     * Vérifie si TensorBoard est initialisé
     */
    public boolean isInitialized() {
        return initialized.get();
    }
    
    /**
     * Obtient l'instance singleton
     */
    public static synchronized TensorBoardExporter getInstance() {
        if (instance == null) {
            instance = new TensorBoardExporter();
        }
        return instance;
    }
    
    /**
     * Initialise TensorBoard avec un répertoire spécifié
     */
    public static boolean initialize(String tensorboardDir) {
        TensorBoardExporter exporter = new TensorBoardExporter(tensorboardDir);
        instance = exporter;
        return exporter.initialize();
    }
    
    /**
     * Crée un StatsListener pour un modèle spécifique
     */
    public static StatsListener getStatsListener(String modelName) {
        if (instance == null) {
            initialize(DEFAULT_TB_DIR);
        }
        return instance.createListener(modelName);
    }
    
    /**
     * Exporte une liste de métriques vers TensorBoard
     */
    public static boolean exportMetrics(List<EvaluationMetrics> metrics, String modelName) {
        if (instance == null) {
            initialize(DEFAULT_TB_DIR);
        }
        
        if (metrics == null || metrics.isEmpty()) {
            logger.warn("Aucune métrique à exporter");
            return false;
        }
        
        logger.info("Exportation de {} métriques pour le modèle {}", metrics.size(), modelName);
        
        // Exporter chaque métrique d'évaluation
        for (EvaluationMetrics metric : metrics) {
            logger.info("Époque {} - {}", metric.getEpoch(), metric);
        }
        
        return true;
    }
    
    /**
     * Arrête le serveur TensorBoard (méthode statique)
     */
    public static void shutdownServer() {
        if (instance != null) {
            instance.shutdown();
        }
    }
}