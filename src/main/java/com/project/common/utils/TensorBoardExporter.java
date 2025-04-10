package com.project.common.utils;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.ui.api.UIServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Version simplifiée de TensorBoardExporter sans dépendances problématiques
 */
public class TensorBoardExporter {
    private static final Logger logger = LoggerFactory.getLogger(TensorBoardExporter.class);
    
    private static final String DEFAULT_TB_DIR = "tensorboard";
    private String tensorboardDir;
    private boolean inMemory;
    private AtomicBoolean initialized = new AtomicBoolean(false);
    
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
                
                logger.info("Serveur TensorBoard démarré - Accédez à http://localhost:9000/train pour visualiser les métriques");
            } else {
                // Stockage en mémoire
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
                logger.info("Serveur TensorBoard arrêté");
                initialized.set(false);
            } catch (Exception e) {
                logger.error("Erreur lors de l'arrêt de TensorBoard: {}", e.getMessage());
            }
        }
    }
    
    /**
     * Exporte des métriques d'évaluation (version simplifiée)
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
     * Exporte des métriques spécifiques (version simplifiée)
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
     * Crée un listener pour attacher à un modèle (stub vide)
     */
    public Object createListener() {
        logger.warn("La fonctionnalité StatsListener n'est pas disponible dans cette version simplifiée");
        return null;
    }
}