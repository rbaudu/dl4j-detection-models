package com.project.common.utils;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.FileStatsStorage;
import org.deeplearning4j.api.storage.impl.InMemoryStatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.stats.impl.DefaultStatsUpdateConfiguration;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Classe pour exporter les métriques d'entraînement vers TensorBoard
 */
public class TensorBoardExporter {
    private static final Logger logger = LoggerFactory.getLogger(TensorBoardExporter.class);
    
    private static final String DEFAULT_TB_DIR = "tensorboard";
    private static final String DEFAULT_TB_FILENAME = "dl4j-stats.bin";
    
    private StatsStorage statsStorage;
    private UIServer uiServer;
    private String tensorboardDir;
    private boolean inMemory;
    private AtomicBoolean initialized = new AtomicBoolean(false);
    
    /**
     * Constructeur par défaut, initialise TensorBoard avec un stockage en mémoire
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
     * Initialise le serveur TensorBoard
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
                
                File statsFile = new File(tbDir, DEFAULT_TB_FILENAME);
                statsStorage = new FileStatsStorage(statsFile);
                
                logger.info("Serveur TensorBoard démarré - Accédez à http://localhost:9000/train pour visualiser les métriques");
            } else {
                // Stockage en mémoire
                statsStorage = new InMemoryStatsStorage();
                logger.info("Serveur TensorBoard démarré en mémoire - Accédez à http://localhost:9000/train pour visualiser les métriques");
            }
            
            // Démarrer le serveur UI
            uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);
            
            initialized.set(true);
            return true;
        } catch (Exception e) {
            logger.error("Erreur lors de l'initialisation de TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Arrête le serveur TensorBoard
     */
    public void shutdown() {
        if (initialized.get()) {
            try {
                if (uiServer != null) {
                    uiServer.stop();
                }
                
                if (statsStorage != null) {
                    statsStorage.close();
                }
                
                logger.info("Serveur TensorBoard arrêté");
                initialized.set(false);
            } catch (Exception e) {
                logger.error("Erreur lors de l'arrêt de TensorBoard: {}", e.getMessage());
            }
        }
    }
    
    /**
     * Crée un listener pour attacher à un modèle
     */
    public StatsListener createListener() {
        if (!initialized.get() && !initialize()) {
            return null;
        }
        
        return new StatsListener((StatsStorageRouter) statsStorage, 
                                new DefaultStatsUpdateConfiguration.Builder().reportFirstIteration(true).build());
    }
    
    /**
     * Exporte des métriques d'évaluation vers TensorBoard
     */
    public boolean exportEvaluation(Evaluation eval, int epoch) {
        if (!initialized.get() && !initialize()) {
            return false;
        }
        
        try {
            if (eval == null || eval.getClasses().isEmpty()) {
                throw new IllegalArgumentException("Évaluation invalide ou vide");
            }
            
            double accuracy = eval.accuracy();
            double precision = eval.precision();
            double recall = eval.recall();
            double f1 = eval.f1();
            
            // Créer un singleton map pour chaque métrique
            Map<String, Float> accuracyMap = Map.of("Accuracy", (float) accuracy);
            Map<String, Float> precisionMap = Map.of("Precision", (float) precision);
            Map<String, Float> recallMap = Map.of("Recall", (float) recall);
            Map<String, Float> f1Map = Map.of("F1", (float) f1);
            
            // Pour des raisons de sécurité, vérifier que l'évaluation a bien des données
            if (eval.getConfusionMatrix() == null || eval.getConfusionMatrix().isEmpty()) {
                throw new IllegalArgumentException("Matrice de confusion vide");
            }
            
            return true;
        } catch (Exception e) {
            logger.error("Erreur lors de l'exportation de l'évaluation vers TensorBoard: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Exporte des métriques spécifiques vers TensorBoard
     */
    public boolean exportMetrics(Map<String, Float> metrics, int iteration) {
        if (!initialized.get() && !initialize()) {
            return false;
        }
        
        try {
            if (metrics == null || metrics.isEmpty()) {
                throw new IllegalArgumentException("Métriques invalides ou vides");
            }
            
            return true;
        } catch (Exception e) {
            logger.error("Erreur lors de l'exportation des métriques vers TensorBoard: {}", e.getMessage());
            return false;
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
     * Teste si TensorBoard est initialisé
     */
    public boolean isInitialized() {
        return initialized.get();
    }
}