# Export des métriques vers TensorBoard

Ce guide explique comment exporter les métriques de vos modèles vers TensorBoard et comment utiliser cet outil pour visualiser l'évolution de l'entraînement.

## Qu'est-ce que TensorBoard ?

TensorBoard est un outil de visualisation conçu à l'origine pour TensorFlow, mais qui peut être utilisé avec d'autres frameworks comme Deeplearning4j. Il permet de visualiser de manière interactive différentes métriques et aspects de l'entraînement de modèles de deep learning, notamment :

- L'évolution des métriques au fil des époques (accuracy, loss, etc.)
- La structure du modèle (graphe de calcul)
- La distribution des poids et des activations
- La visualisation des embeddings
- Et bien d'autres fonctionnalités

## Configuration de TensorBoard

Pour installer et configurer TensorBoard, consultez le guide [Installation et configuration de TensorBoard](TENSORBOARD_INSTALL.md).

### Paramètres dans application.properties

L'exportation vers TensorBoard est configurée dans le fichier `application.properties` :

```properties
# Configuration TensorBoard
tensorboard.enabled=true
tensorboard.log.dir=output/tensorboard
tensorboard.port=6006
tensorboard.export.epoch.frequency=1
```

- `tensorboard.enabled` : Active ou désactive l'exportation vers TensorBoard
- `tensorboard.log.dir` : Répertoire où les logs TensorBoard seront stockés
- `tensorboard.port` : Port sur lequel TensorBoard sera accessible
- `tensorboard.export.epoch.frequency` : Fréquence d'exportation des métriques (1 = à chaque époque)

## Architecture d'exportation vers TensorBoard

### Classe TensorBoardExporter

La classe `TensorBoardExporter` est responsable de l'exportation des métriques vers TensorBoard. Cette classe :

1. Crée un répertoire spécifique pour chaque modèle et exécution
2. Configure l'écrivain de logs TensorBoard
3. Exporte les métriques d'entraînement et d'évaluation
4. Gère la fermeture propre des ressources

Voici un extrait de son implémentation :

```java
public class TensorBoardExporter implements Closeable {
    private final String logDir;
    private final SummaryWriter writer;
    private final String modelName;
    
    public TensorBoardExporter(String modelName, Properties config) {
        this.modelName = modelName;
        this.logDir = config.getProperty("tensorboard.log.dir", "output/tensorboard") 
            + "/" + modelName + "/" + System.currentTimeMillis();
        
        // Création du répertoire s'il n'existe pas
        new File(logDir).mkdirs();
        
        // Initialisation de l'écrivain TensorBoard
        this.writer = new SummaryWriter(logDir);
    }
    
    // Exporte les métriques d'un epoch
    public void exportEpochMetrics(EvaluationMetrics metrics, int epoch) {
        // Exporte les métriques principales
        writer.addScalar("accuracy", metrics.getAccuracy(), epoch);
        writer.addScalar("precision", metrics.getPrecision(), epoch);
        writer.addScalar("recall", metrics.getRecall(), epoch);
        writer.addScalar("f1", metrics.getF1Score(), epoch);
        writer.addScalar("loss", metrics.getLoss(), epoch);
        
        // Exporte des métriques par classe si disponibles
        if (metrics.getPerClassMetrics() != null) {
            for (int i = 0; i < metrics.getPerClassMetrics().length; i++) {
                PerClassMetric classMetric = metrics.getPerClassMetrics()[i];
                String className = "class_" + i;
                
                writer.addScalar("class/" + className + "/precision", classMetric.getPrecision(), epoch);
                writer.addScalar("class/" + className + "/recall", classMetric.getRecall(), epoch);
                writer.addScalar("class/" + className + "/f1", classMetric.getF1Score(), epoch);
            }
        }
        
        // Flush des données pour assurer l'écriture
        writer.flush();
    }
    
    // Exporte la matrice de confusion
    public void exportConfusionMatrix(INDArray confusionMatrix, int epoch) {
        // Conversion en image pour visualisation dans TensorBoard
        BufferedImage confusionImage = convertConfusionMatrixToImage(confusionMatrix);
        writer.addImage("confusion_matrix", confusionImage, epoch);
        writer.flush();
    }
    
    // Fermeture des ressources
    @Override
    public void close() {
        if (writer != null) {
            try {
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    
    // Méthode utilitaire pour convertir une matrice de confusion en image
    private BufferedImage convertConfusionMatrixToImage(INDArray matrix) {
        // Implémentation de la conversion...
    }
}
```

## Intégration avec le système de métriques

L'exportation vers TensorBoard est intégrée dans la classe `MetricsTracker` :

```java
public class MetricsTracker {
    private final List<EvaluationMetrics> metrics = new ArrayList<>();
    private TensorBoardExporter tensorBoardExporter;
    private final boolean tensorBoardEnabled;
    private final int exportFrequency;
    
    public MetricsTracker(String modelName, Properties config) {
        // Vérification si TensorBoard est activé
        tensorBoardEnabled = Boolean.parseBoolean(
            config.getProperty("tensorboard.enabled", "false"));
        
        // Fréquence d'exportation
        exportFrequency = Integer.parseInt(
            config.getProperty("tensorboard.export.epoch.frequency", "1"));
        
        // Initialisation de l'exportateur TensorBoard si activé
        if (tensorBoardEnabled) {
            tensorBoardExporter = new TensorBoardExporter(modelName, config);
        }
    }
    
    // Ajoute des métriques et les exporte vers TensorBoard si activé
    public void addMetrics(EvaluationMetrics newMetrics) {
        metrics.add(newMetrics);
        
        // Exportation vers TensorBoard si activé et selon la fréquence configurée
        if (tensorBoardEnabled && newMetrics.getEpoch() % exportFrequency == 0) {
            tensorBoardExporter.exportEpochMetrics(newMetrics, newMetrics.getEpoch());
            
            // Exporter la matrice de confusion si disponible
            if (newMetrics.getConfusionMatrix() != null) {
                tensorBoardExporter.exportConfusionMatrix(
                    newMetrics.getConfusionMatrix(), newMetrics.getEpoch());
            }
        }
    }
    
    // Fermeture propre des ressources
    public void close() {
        if (tensorBoardEnabled && tensorBoardExporter != null) {
            try {
                tensorBoardExporter.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    
    // Autres méthodes...
}
```

## Utilisation de TensorBoard

### Lancement de TensorBoard

Pour visualiser les logs exportés, lancez TensorBoard en pointant vers le répertoire de logs :

```bash
tensorboard --logdir=output/tensorboard --port=6006
```

Par défaut, TensorBoard sera accessible à l'adresse [http://localhost:6006](http://localhost:6006).

### Interface de TensorBoard

Une fois TensorBoard lancé, vous pouvez accéder à plusieurs onglets :

1. **Scalars** : Visualisation des métriques numériques au fil du temps (accuracy, loss, etc.)

   <!-- Ajouter une capture d'écran réelle de TensorBoard ici -->
   <!-- ![TensorBoard Scalars](images/tensorboard_scalars.png) -->
   <!-- *Exemple de visualisation des métriques d'accuracy, loss, precision et recall au fil des époques* -->

2. **Images** : Visualisation des matrices de confusion exportées

   <!-- Ajouter une capture d'écran réelle de TensorBoard ici -->
   <!-- ![TensorBoard Images](images/tensorboard_confusion_matrix.png) -->
   <!-- *Exemple de matrice de confusion visualisée dans TensorBoard* -->

3. **Graphs** : Visualisation de la structure du modèle (si exportée)

   <!-- Ajouter une capture d'écran réelle de TensorBoard ici -->
   <!-- ![TensorBoard Graphs](images/tensorboard_graph.png) -->
   <!-- *Représentation graphique d'un modèle VGG16 dans TensorBoard* -->

4. **Distributions** : Visualisation de la distribution des poids (si exportée)
5. **Histograms** : Histogrammes des poids et activations (si exportés)

### Visualisation de plusieurs modèles

TensorBoard permet de comparer facilement plusieurs modèles ou exécutions. Il suffit de pointer vers le répertoire parent contenant les logs de tous les modèles :

```bash
tensorboard --logdir=output/tensorboard
```

Les différents modèles et exécutions seront automatiquement regroupés et pourront être comparés dans l'interface.

<!-- Ajouter une capture d'écran réelle de TensorBoard ici -->
<!-- ![TensorBoard Model Comparison](images/tensorboard_comparison.png) -->
<!-- *Comparaison des performances de trois modèles (VGG16, ResNet et MobileNet) sur les mêmes données* -->

## Exemple d'utilisation

Voici un exemple d'utilisation de l'exportation vers TensorBoard lors de l'entraînement d'un modèle :

```java
// Chargement de la configuration
Properties config = loadConfiguration("config/application.properties");

// Vérification que TensorBoard est activé
boolean tensorBoardEnabled = Boolean.parseBoolean(config.getProperty("tensorboard.enabled", "false"));
System.out.println("TensorBoard export " + (tensorBoardEnabled ? "enabled" : "disabled"));

// Création et entraînement du modèle
ActivityTrainer trainer = new ActivityTrainer(config);
trainer.train();

// Les métriques sont automatiquement exportées vers TensorBoard pendant l'entraînement
// si l'option est activée dans la configuration

// Fermeture propre du tracker de métriques pour assurer l'écriture des logs
trainer.getMetricsTracker().close();

// Information pour l'utilisateur
if (tensorBoardEnabled) {
    String tensorboardDir = config.getProperty("tensorboard.log.dir", "output/tensorboard");
    int tensorboardPort = Integer.parseInt(config.getProperty("tensorboard.port", "6006"));
    System.out.println("TensorBoard logs saved to: " + tensorboardDir);
    System.out.println("Launch TensorBoard with: tensorboard --logdir=" + tensorboardDir + " --port=" + tensorboardPort);
    System.out.println("Then open http://localhost:" + tensorboardPort + " in your browser");
}
```

## Avantages de l'utilisation de TensorBoard

L'utilisation de TensorBoard offre plusieurs avantages par rapport aux visualisations statiques :

1. **Interactivité** : Vous pouvez zoomer, filtrer et explorer les données de manière interactive
2. **Comparaison facile** : Comparez facilement plusieurs modèles ou exécutions côte à côte
3. **Visualisations riches** : Accédez à des visualisations plus avancées comme les graphes de modèles, les distributions, etc.
4. **Temps réel** : Visualisez les métriques en temps réel pendant l'entraînement
5. **Partage simplifié** : Les logs peuvent être facilement partagés avec d'autres membres de l'équipe

## Notes sur les captures d'écran

Pour des exemples visuels de l'interface TensorBoard, consultez le répertoire `docs/images/`. Vous pouvez y trouver des exemples des différentes vues de TensorBoard qui vous aideront à comprendre comment interpréter les visualisations.

Pour ajouter vos propres captures d'écran, suivez les instructions dans le fichier `docs/images/tensorboard_placeholder.md`.
