# Système de métriques d'évaluation

Le projet intègre un système complet de métriques d'évaluation qui permet de :

1. **Collecter les métriques pendant l'entraînement** - Suivi automatique des métriques (accuracy, precision, recall, F1-score)
2. **Générer des visualisations graphiques** - Création de graphiques d'évolution des métriques et comparaisons
3. **Produire des rapports détaillés** - Génération de rapports au format texte et CSV
4. **Comparer différents modèles** - Outils pour évaluer objectivement les performances de différentes architectures
5. **Exporter les métriques vers TensorBoard** - Visualisation interactive des métriques (voir [TensorBoard](TENSORBOARD.md))

## Classes principales du système de métriques

- **EvaluationMetrics** - Stocke les métriques d'évaluation
- **MetricsTracker** - Collecte automatiquement les métriques pendant l'entraînement
- **MetricsVisualizer** - Génère des visualisations graphiques des métriques
- **ModelEvaluator** - Produit des rapports d'évaluation détaillés
- **MetricsUtils** - Fournit des méthodes utilitaires pour faciliter l'utilisation
- **TensorBoardExporter** - Exporte les métriques vers TensorBoard (voir [TensorBoard](TENSORBOARD.md))

## Utilisation du système de métriques d'évaluation

Le système de métriques peut être utilisé de plusieurs façons :

### 1. Métriques automatiques pendant l'entraînement

La classe `ModelTrainer` collecte automatiquement les métriques pendant l'entraînement :

```java
// Créer un entraîneur
ActivityTrainer trainer = new ActivityTrainer(config);

// Entraîner le modèle (les métriques sont collectées automatiquement)
trainer.train();

// Accéder aux métriques collectées
MetricsTracker tracker = trainer.getMetricsTracker();
List<EvaluationMetrics> metrics = tracker.getMetrics();

// Visualiser les métriques
MetricsVisualizer.generateAllCharts(metrics, "output/metrics", "activity_model");
```

### 2. Évaluation ponctuelle d'un modèle

Vous pouvez évaluer un modèle existant à tout moment :

```java
// Charger un modèle
ActivityModel model = new ActivityModel(config);
model.loadDefaultModel();

// Évaluer le modèle
EvaluationMetrics metrics = MetricsUtils.evaluateModel(
    model.getNetwork(), testData, "activity_model", config);

// Vérifier les performances par rapport aux seuils
boolean valid = MetricsUtils.validateMetrics(metrics, config);
```

### 3. Comparaison de modèles

```java
// Générer un rapport comparatif
MetricsUtils.generateModelComparisonReport(
    new EvaluationMetrics[] { vgg16Metrics, resnetMetrics, mobileNetMetrics },
    new String[] { "VGG16", "ResNet", "MobileNet" },
    "output/metrics/model_comparison.txt"
);
```

### 4. Génération de visualisations

```java
// Générer un graphique d'évolution de la précision
MetricsVisualizer.generateAccuracyChart(metricsList, "output/metrics/charts", "model_accuracy");

// Générer un graphique d'évolution du F1-score
MetricsVisualizer.generateF1Chart(metricsList, "output/metrics/charts", "model_f1");

// Générer tous les graphiques
MetricsVisualizer.generateAllCharts(metricsList, "output/metrics/charts", "model");
```

### 5. Exportation des métriques vers CSV

```java
// Exporter les métriques vers un fichier CSV
MetricsUtils.exportMetricsToCSV(metricsList, "output/metrics/csv/metrics_export.csv");
```

## Métriques disponibles

Le système collecte et gère les métriques suivantes :

- **Accuracy** - Pourcentage global de prédictions correctes
- **Precision** - Ratio des vrais positifs par rapport à tous les positifs prédits (par classe et global)
- **Recall** - Ratio des vrais positifs par rapport à tous les positifs réels (par classe et global)
- **F1-Score** - Moyenne harmonique de la précision et du rappel (par classe et global)
- **Loss** - Valeur de la fonction de perte pendant l'entraînement
- **Confusion Matrix** - Matrice de confusion pour analyser les erreurs de classification

## Configuration des métriques

Vous pouvez configurer le système de métriques via le fichier `application.properties` :

```properties
# Configuration des métriques d'évaluation
metrics.output.dir=output/metrics
test.min.accuracy=0.8
test.min.precision=0.75
test.min.recall=0.75
test.min.f1=0.75
evaluation.batch.size=32
metrics.evaluation.frequency=1
```

Pour configurer l'exportation vers TensorBoard, consultez le [guide TensorBoard](TENSORBOARD.md).

Pour des exemples complets, consultez la classe `MetricsExampleUsage.java`.
