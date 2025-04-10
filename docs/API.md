# Documentation de l'API

Ce document fournit une documentation détaillée de l'API du projet avec des exemples de code complets.

## Table des matières

1. [API ModelManager](#api-modelmanager)
2. [API des modèles](#api-des-modèles)
   - [Modèles de détection de présence](#modèles-de-détection-de-présence)
   - [Modèles de détection d'activité](#modèles-de-détection-dactivité)
   - [Modèles de détection de sons](#modèles-de-détection-de-sons)
3. [API des métriques](#api-des-métriques)
4. [API TensorBoard](#api-tensorboard)

## API ModelManager

### Initialisation

```java
// Chargement de la configuration
Properties config = new Properties();
try (InputStream is = new FileInputStream("config/application.properties")) {
    config.load(is);
} catch (Exception e) {
    throw new RuntimeException("Impossible de charger la configuration", e);
}

// Création du gestionnaire de modèles
ModelManager modelManager = new ModelManager(config);
```

### Sélection des modèles

```java
// Définir le type de modèle de présence
modelManager.setPresenceModelType(ModelManager.PresenceModelType.YOLO);

// Définir le type de modèle d'activité
modelManager.setActivityModelType(ModelManager.ActivityModelType.VGG16);

// Définir le type de modèle de son
modelManager.setSoundModelType(ModelManager.SoundModelType.SPECTROGRAM);
```

### Récupération des modèles

```java
// Récupérer le modèle de présence approprié
BasePresenceModel presenceModel = modelManager.getPresenceModel();
// ou spécifiquement le modèle YOLO
YOLOPresenceModel yoloModel = modelManager.getYoloPresenceModel();

// Récupérer le modèle d'activité approprié
BaseActivityModel activityModel = modelManager.getActivityModel();
// ou spécifiquement le modèle VGG16
VGG16ActivityModel vgg16Model = modelManager.getVgg16ActivityModel();

// Récupérer le modèle de son approprié
BaseSoundModel soundModel = modelManager.getSoundModel();
// ou spécifiquement le modèle basé sur spectrogrammes
SpectrogramSoundModel spectrogramModel = modelManager.getSpectrogramSoundModel();
```

### Exemple complet avec traitement d'image

```java
// Initialisation
Properties config = loadConfiguration("config/application.properties");
ModelManager modelManager = new ModelManager(config);
modelManager.setActivityModelType(ModelManager.ActivityModelType.VGG16);

// Récupération du modèle
VGG16ActivityModel model = modelManager.getVgg16ActivityModel();

// Chargement d'une image
File imageFile = new File("data/test/activity_test_image.jpg");
BufferedImage image = ImageIO.read(imageFile);

// Prétraitement de l'image
INDArray preprocessedImage = model.preprocessImage(image);

// Prédiction
int predictedClassIndex = model.predictActivity(preprocessedImage);
String predictedClassName = model.getClassNameFromIndex(predictedClassIndex);

System.out.println("Activité détectée : " + predictedClassName);
```

## API des modèles

### Modèles de détection de présence

#### YOLOPresenceModel

```java
// Initialisation
YOLOPresenceModel yoloModel = new YOLOPresenceModel(config);

// Chargement d'un modèle pré-entraîné
yoloModel.loadModel("models/presence/yolo_model.zip");

// Détection de présence dans une image
BufferedImage image = ImageIO.read(new File("data/test/presence_test.jpg"));
boolean presenceDetected = yoloModel.detectPresence(image);

// Détection avec informations détaillées
DetectionResult result = yoloModel.detectWithDetails(image);
System.out.println("Présence détectée : " + result.isPresenceDetected());
System.out.println("Confiance : " + result.getConfidence());
System.out.println("Boîtes englobantes : " + result.getBoundingBoxes().size());

// Accès aux boîtes englobantes
for (BoundingBox box : result.getBoundingBoxes()) {
    System.out.println("Classe : " + box.getLabel());
    System.out.println("Confiance : " + box.getConfidence());
    System.out.println("Position : (" + box.getX() + ", " + box.getY() + ")");
    System.out.println("Dimensions : " + box.getWidth() + " x " + box.getHeight());
}
```

### Modèles de détection d'activité

#### VGG16ActivityModel

```java
// Initialisation et chargement
VGG16ActivityModel model = new VGG16ActivityModel(config);
model.loadModel("models/activity/vgg16_model.zip");

// Prédiction sur une image
BufferedImage image = ImageIO.read(new File("data/test/activity_test.jpg"));
int classIndex = model.predictActivity(image);
String className = model.getClassNameFromIndex(classIndex);

// Prédiction avec probabilités pour toutes les classes
INDArray probabilities = model.predictProbabilities(image);
Map<String, Double> classConfidences = new HashMap<>();

for (int i = 0; i < model.getNumClasses(); i++) {
    String name = model.getClassNameFromIndex(i);
    double probability = probabilities.getDouble(i);
    classConfidences.put(name, probability);
}

// Affichage des 3 classes les plus probables
classConfidences.entrySet().stream()
    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
    .limit(3)
    .forEach(entry -> System.out.println(entry.getKey() + ": " + 
                                        String.format("%.2f%%", entry.getValue() * 100)));
```

### Modèles de détection de sons

#### SpectrogramSoundModel

```java
// Initialisation
SpectrogramSoundModel model = new SpectrogramSoundModel(config);
model.loadModel("models/sound/spectrogram_model.zip");

// Prédiction simple sur un fichier audio
String audioFilePath = "data/test/sound_test.wav";
String predictedClass = model.predict(audioFilePath);
System.out.println("Activité sonore détectée : " + predictedClass);

// Génération et visualisation d'un spectrogramme
BufferedImage spectrogram = model.generateSpectrogram(audioFilePath);
ImageIO.write(spectrogram, "PNG", new File("output/spectrograms/test_spectrogram.png"));

// Prédiction avec probabilités détaillées
Map<String, Double> soundProbabilities = model.predictWithProbabilities(audioFilePath);
soundProbabilities.entrySet().stream()
    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
    .limit(3)
    .forEach(entry -> System.out.println(entry.getKey() + ": " + 
                                        String.format("%.2f%%", entry.getValue() * 100)));
```

## API des métriques

### Évaluation d'un modèle

```java
// Charger un modèle
VGG16ActivityModel model = new VGG16ActivityModel(config);
model.loadModel("models/activity/vgg16_model.zip");

// Préparer un jeu de données de test
DataSetIterator testIterator = new RecordReaderDataSetIterator(
    new ImageRecordReader(height, width, channels),
    batchSize, labelIndex, numClasses);

// Évaluer le modèle
EvaluationMetrics metrics = MetricsUtils.evaluateModel(
    model.getNetwork(), testIterator, "vgg16_model", config);

// Afficher les résultats
System.out.println("Accuracy: " + metrics.getAccuracy());
System.out.println("Precision: " + metrics.getPrecision());
System.out.println("Recall: " + metrics.getRecall());
System.out.println("F1 Score: " + metrics.getF1Score());

// Générer un rapport d'évaluation détaillé
String reportPath = "output/metrics/reports/vgg16_evaluation_report.txt";
MetricsUtils.generateEvaluationReport(metrics, reportPath);

// Valider les performances par rapport aux seuils configurés
boolean isValid = MetricsUtils.validateMetrics(metrics, config);
if (isValid) {
    System.out.println("Le modèle répond aux critères de performance minimaux");
} else {
    System.out.println("Le modèle ne répond pas aux critères de performance");
    // Afficher les métriques spécifiques qui ne satisfont pas les seuils
    if (metrics.getAccuracy() < Double.parseDouble(config.getProperty("test.min.accuracy"))) {
        System.out.println("L'accuracy est trop faible");
    }
    // etc.
}
```

### Visualisation des métriques

```java
// Charger des métriques précédemment collectées
List<EvaluationMetrics> metricsList = loadMetricsFromFile("output/metrics/saved_metrics.ser");

// Générer tous les graphiques
String outputDir = "output/metrics/charts";
String baseFileName = "vgg16_model";
MetricsVisualizer.generateAllCharts(metricsList, outputDir, baseFileName);

// Ou générer des graphiques spécifiques
MetricsVisualizer.generateAccuracyChart(metricsList, outputDir, baseFileName + "_accuracy");
MetricsVisualizer.generateLossChart(metricsList, outputDir, baseFileName + "_loss");
MetricsVisualizer.generatePrecisionRecallChart(metricsList, outputDir, baseFileName + "_pr");
MetricsVisualizer.generateF1Chart(metricsList, outputDir, baseFileName + "_f1");

// Générer un graphique personnalisé
JFreeChart customChart = MetricsVisualizer.createCustomChart(
    metricsList,
    "Évolution de Performance",
    "Époques",
    "Valeur",
    metrics -> new double[]{metrics.getAccuracy(), metrics.getPrecision()},
    new String[]{"Accuracy", "Precision"}
);
MetricsVisualizer.saveChartAsPNG(customChart, outputDir + "/custom_chart.png", 800, 600);
```

### Comparaison de modèles

```java
// Évaluer plusieurs modèles
EvaluationMetrics vgg16Metrics = evaluateModel(vgg16Model, testData, "vgg16", config);
EvaluationMetrics resnetMetrics = evaluateModel(resnetModel, testData, "resnet", config);
EvaluationMetrics mobileNetMetrics = evaluateModel(mobileNetModel, testData, "mobilenet", config);

// Générer un rapport comparatif
String comparisonReportPath = "output/metrics/reports/model_comparison.txt";
MetricsUtils.generateModelComparisonReport(
    new EvaluationMetrics[]{vgg16Metrics, resnetMetrics, mobileNetMetrics},
    new String[]{"VGG16", "ResNet", "MobileNet"},
    comparisonReportPath
);

// Générer des graphiques comparatifs
MetricsVisualizer.generateComparisonChart(
    new EvaluationMetrics[][]{
        vgg16MetricsList.toArray(new EvaluationMetrics[0]),
        resnetMetricsList.toArray(new EvaluationMetrics[0]),
        mobileNetMetricsList.toArray(new EvaluationMetrics[0])
    },
    new String[]{"VGG16", "ResNet", "MobileNet"},
    "Comparaison de l'accuracy",
    "Époques",
    "Accuracy",
    metrics -> metrics.getAccuracy(),
    "output/metrics/charts/accuracy_comparison.png"
);
```

## API TensorBoard

### Initialisation et utilisation de l'exportateur TensorBoard

```java
// Initialisation de l'exportateur
TensorBoardExporter tensorBoard = new TensorBoardExporter("vgg16_model", config);

// Exportation des métriques à chaque époque pendant l'entraînement
for (int epoch = 0; epoch < numEpochs; epoch++) {
    // Entraînement du modèle...
    
    // Évaluer le modèle pour obtenir les métriques
    EvaluationMetrics metrics = evaluateModel(model, validationData);
    
    // Exporter les métriques vers TensorBoard
    tensorBoard.exportEpochMetrics(metrics, epoch);
    
    // Exporter la matrice de confusion si disponible
    if (metrics.getConfusionMatrix() != null) {
        tensorBoard.exportConfusionMatrix(metrics.getConfusionMatrix(), epoch);
    }
}

// Fermeture de l'exportateur à la fin de l'entraînement
tensorBoard.close();
```

### Intégration avec le système d'entraînement

```java
public class ModelTrainer {
    private ComputationGraph model;
    private DataSetIterator trainData;
    private DataSetIterator testData;
    private MetricsTracker metricsTracker;
    private TensorBoardExporter tensorBoard;
    private Properties config;
    
    public ModelTrainer(String modelName, Properties config) {
        this.config = config;
        
        // Initialisation du tracker de métriques (qui initialise aussi TensorBoard si activé)
        this.metricsTracker = new MetricsTracker(modelName, config);
        
        // Initialisation directe de TensorBoard si nécessaire
        boolean tensorBoardEnabled = Boolean.parseBoolean(config.getProperty("tensorboard.enabled", "false"));
        if (tensorBoardEnabled) {
            this.tensorBoard = new TensorBoardExporter(modelName, config);
        }
    }
    
    public void train(int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Entraînement pour une époque
            trainOneEpoch();
            
            // Évaluation du modèle
            EvaluationMetrics metrics = evaluateModel();
            
            // Ajouter les métriques au tracker (qui les exporte vers TensorBoard si activé)
            metricsTracker.addMetrics(metrics);
            
            // Alternative : exporter directement vers TensorBoard
            if (tensorBoard != null) {
                tensorBoard.exportEpochMetrics(metrics, epoch);
            }
        }
        
        // Fermeture de TensorBoard
        if (tensorBoard != null) {
            try {
                tensorBoard.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    
    // Autres méthodes...
}
```

### Exportation avancée vers TensorBoard

```java
// Exportation personnalisée de données vers TensorBoard
public class CustomTensorBoardExporter {
    private SummaryWriter writer;
    private String logDir;
    
    public CustomTensorBoardExporter(String modelName, Properties config) {
        this.logDir = config.getProperty("tensorboard.log.dir", "output/tensorboard") 
            + "/" + modelName + "/" + System.currentTimeMillis();
        new File(logDir).mkdirs();
        this.writer = new SummaryWriter(logDir);
    }
    
    // Exporter la structure du modèle (graphe de calcul)
    public void exportModelGraph(ComputationGraph model) {
        try {
            // Exporter le graphe du modèle au format protobuf
            String graphPath = logDir + "/model_graph.pb";
            model.save(new File(graphPath), true);
            
            // Ajouter le graphe à TensorBoard
            writer.addGraph(model);
        } catch (Exception e) {
            System.err.println("Erreur lors de l'exportation du graphe du modèle : " + e.getMessage());
        }
    }
    
    // Exporter des histogrammes de poids
    public void exportWeightHistograms(ComputationGraph model, int step) {
        Map<String, INDArray> params = model.paramTable();
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            String paramName = entry.getKey();
            INDArray paramValues = entry.getValue();
            writer.addHistogram(paramName, paramValues, step);
        }
    }
    
    // Exporter des embeddings (représentations vectorielles)
    public void exportEmbeddings(INDArray embeddings, List<String> labels, BufferedImage[] images, String metadataFileName) {
        // Écrire les métadonnées (labels)
        try {
            File metadataFile = new File(logDir, metadataFileName);
            try (PrintWriter writer = new PrintWriter(metadataFile)) {
                writer.println("Name\tClass");
                for (int i = 0; i < labels.size(); i++) {
                    writer.println("data_" + i + "\t" + labels.get(i));
                }
            }
            
            // Ajouter les embeddings à TensorBoard
            this.writer.addEmbedding(embeddings, metadataFile, images);
        } catch (Exception e) {
            System.err.println("Erreur lors de l'exportation des embeddings : " + e.getMessage());
        }
    }
    
    public void close() {
        if (writer != null) {
            try {
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```
