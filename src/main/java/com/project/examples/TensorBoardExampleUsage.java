package com.project.examples;

import com.project.common.config.ConfigLoader;
import com.project.common.utils.EvaluationMetrics;
import com.project.common.utils.MetricsTracker;
import com.project.common.utils.TensorBoardExporter;
import com.project.models.activity.ActivityModel;
import com.project.training.ActivityTrainer;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * Exemple d'utilisation de TensorBoard pour la visualisation des métriques.
 * Cette classe montre comment configurer et utiliser TensorBoard avec les modèles DL4J.
 */
public class TensorBoardExampleUsage {
    private static final Logger log = LoggerFactory.getLogger(TensorBoardExampleUsage.class);
    
    public static void main(String[] args) {
        log.info("Démarrage de l'exemple d'utilisation de TensorBoard");
        
        try {
            // Charger la configuration
            Properties config = ConfigLoader.loadConfiguration();
            
            // Configurer le répertoire de sortie pour les logs TensorBoard
            String tensorBoardDir = "output/tensorboard";
            config.setProperty("tensorboard.log.dir", tensorBoardDir);
            config.setProperty("tensorboard.enabled", "true");
            
            // Exemple 1: Utilisation directe de TensorBoard pendant l'entraînement
            directTensorBoardUsage(config);
            
            // Exemple 2: Exportation des métriques existantes vers TensorBoard
            exportExistingMetricsToTensorBoard(config);
            
            // Exemple 3: Utilisation avec MetricsTracker
            useWithMetricsTracker(config);
            
            // Arrêter proprement TensorBoard
            TensorBoardExporter.shutdown();
            
            log.info("Exemple d'utilisation de TensorBoard terminé avec succès");
            log.info("Pour visualiser les métriques, lancez TensorBoard en pointant vers le répertoire {} :", tensorBoardDir);
            log.info("tensorboard --logdir={}", tensorBoardDir);
            log.info("Puis ouvrez http://localhost:6006 dans votre navigateur");
            
        } catch (Exception e) {
            log.error("Erreur lors de l'exécution de l'exemple", e);
        }
    }
    
    /**
     * Exemple 1: Utilisation directe de TensorBoard pendant l'entraînement
     */
    private static void directTensorBoardUsage(Properties config) throws IOException {
        log.info("=== Exemple 1: Utilisation directe de TensorBoard pendant l'entraînement ===");
        
        // Initialiser TensorBoard avec un répertoire de logs
        String tensorBoardDir = config.getProperty("tensorboard.log.dir", "output/tensorboard");
        TensorBoardExporter.initialize(tensorBoardDir);
        
        // Charger ou créer un modèle
        ActivityModel model = new ActivityModel(config);
        model.initializeModel();
        
        // Préparer des données fictives pour l'exemple
        ActivityTrainer trainer = new ActivityTrainer(config);
        DataSet[] datasets = trainer.prepareData("data/raw/activity", 0.8);
        DataSet trainData = datasets[0];
        DataSet testData = datasets[1];
        
        // Créer des itérateurs
        DataSetIterator trainIterator = new ListDataSetIterator<>(
            Collections.singletonList(trainData), 32);
        DataSetIterator testIterator = new ListDataSetIterator<>(
            Collections.singletonList(testData), 32);
        
        // Obtenir un StatsListener de TensorBoard pour suivre les métriques d'entraînement
        StatsListener statsListener = TensorBoardExporter.getStatsListener("activity_model_direct");
        
        // Ajouter le StatsListener au modèle
        model.getNetwork().setListeners(statsListener);
        
        // Simuler un entraînement simple (3 époques pour l'exemple)
        log.info("Simulation d'un entraînement avec suivi TensorBoard en direct...");
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.getNetwork().fit(trainIterator);
            trainIterator.reset();
            
            // Évaluer le modèle après chaque époque
            model.getNetwork().evaluate(testIterator);
            testIterator.reset();
            
            // Exporter explicitement l'évaluation vers TensorBoard (facultatif, déjà fait par le StatsListener)
            model.getNetwork().evaluate(testIterator);
            testIterator.reset();
        }
        
        log.info("Entraînement terminé, métriques disponibles dans TensorBoard");
    }
    
    /**
     * Exemple 2: Exportation des métriques existantes vers TensorBoard
     */
    private static void exportExistingMetricsToTensorBoard(Properties config) {
        log.info("=== Exemple 2: Exportation des métriques existantes vers TensorBoard ===");
        
        // Créer quelques métriques factices pour l'exemple
        List<EvaluationMetrics> metricsList = MetricsExampleUsage.createSampleMetrics();
        
        // Exporter les métriques vers TensorBoard
        String tensorBoardDir = config.getProperty("tensorboard.log.dir", "output/tensorboard");
        boolean success = TensorBoardExporter.initialize(tensorBoardDir);
        
        if (success) {
            TensorBoardExporter.exportMetrics(metricsList, "existing_metrics");
            log.info("Métriques existantes exportées vers TensorBoard");
        } else {
            log.error("Échec de l'initialisation de TensorBoard pour l'export");
        }
    }
    
    /**
     * Exemple 3: Utilisation avec MetricsTracker
     */
    private static void useWithMetricsTracker(Properties config) throws IOException {
        log.info("=== Exemple 3: Utilisation avec MetricsTracker ===");
        
        // Créer un modèle pour l'exemple
        ActivityModel model = new ActivityModel(config);
        model.initializeModel();
        
        // Préparer des données fictives pour l'exemple
        ActivityTrainer trainer = new ActivityTrainer(config);
        DataSet[] datasets = trainer.prepareData("data/raw/activity", 0.8);
        DataSet trainData = datasets[0];
        DataSet testData = datasets[1];
        
        // Créer des itérateurs
        DataSetIterator trainIterator = new ListDataSetIterator<>(
            Collections.singletonList(trainData), 32);
        DataSetIterator testIterator = new ListDataSetIterator<>(
            Collections.singletonList(testData), 32);
        
        // Créer un MetricsTracker avec export TensorBoard activé
        MetricsTracker tracker = new MetricsTracker(
            testIterator, 
            1, 
            "output/metrics", 
            "activity_model_tracker", 
            true  // Activer TensorBoard
        );
        
        // Ajouter le tracker comme listener au modèle
        model.getNetwork().setListeners(tracker);
        
        // Simuler un entraînement (3 époques pour l'exemple)
        log.info("Simulation d'un entraînement avec MetricsTracker et TensorBoard...");
        for (int epoch = 1; epoch <= 3; epoch++) {
            model.getNetwork().setEpochCount(epoch);  // Important pour le tracker
            model.getNetwork().fit(trainIterator);
            trainIterator.reset();
            
            // Le tracker exportera automatiquement les métriques vers TensorBoard
        }
        
        log.info("Entraînement terminé, métriques disponibles dans TensorBoard");
    }
    
    /**
     * Utilisation standard de TensorBoard avec DL4J (méthode standard sans nos classes utilitaires)
     * Cette méthode est fournie à titre éducatif pour montrer comment fonctionne
     * TensorBoard en standard avec DL4J.
     */
    private static void standardTensorBoardUsage(Properties config) {
        log.info("=== Bonus: Utilisation standard de TensorBoard avec DL4J ===");
        
        try {
            // Initialiser le serveur UI
            UIServer uiServer = UIServer.getInstance();
            
            // Choisir un stockage pour les statistiques (fichier ou mémoire)
            // Option 1: Stockage en fichier (persistant)
            String logDir = config.getProperty("tensorboard.log.dir", "output/tensorboard");
            FileStatsStorage statsStorage = new FileStatsStorage(new File(logDir, "ui-stats.bin"));
            
            // Option 2: Stockage en mémoire (non persistant)
            // InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
            
            // Attacher le stockage au serveur UI
            uiServer.attach(statsStorage);
            
            // Créer un modèle
            ActivityModel model = new ActivityModel(config);
            model.initializeModel();
            
            // Ajouter un StatsListener pour collecter les statistiques
            model.getNetwork().setListeners(new StatsListener(statsStorage));
            
            // Ici, vous pourriez entraîner le modèle comme d'habitude
            // model.getNetwork().fit(trainData);
            
            log.info("Serveur UI démarré. Accédez à http://localhost:9000/train pour voir les statistiques");
            
        } catch (Exception e) {
            log.error("Erreur lors de l'initialisation de UI standard: {}", e.getMessage());
        }
    }
}
