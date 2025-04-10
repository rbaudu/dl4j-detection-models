# Utilisation avancée et personnalisation

Ce guide explique comment personnaliser et étendre les fonctionnalités du projet pour des cas d'usage avancés.

## Personnalisation des modèles

### Modification des hyperparamètres

Les hyperparamètres des modèles peuvent être ajustés dans le fichier `config/application.properties`. Voici quelques paramètres importants que vous pourriez vouloir modifier :

```properties
# Hyperparamètres généraux
training.seed=123
training.use.regularization=true
training.l2=0.0001
training.dropout=0.5
training.updater=adam

# Hyperparamètres spécifiques aux modèles
activity.model.learning.rate=0.0005
activity.model.batch.size=64
activity.model.epochs=150
activity.model.dropout=0.5

presence.model.learning.rate=0.001
sound.model.learning.rate=0.0001
```

### Modification des architectures

Pour personnaliser l'architecture des modèles au-delà des paramètres de configuration, vous devrez modifier les classes correspondantes dans le package `models`. Voici quelques points d'extension courants :

#### Exemple : Personnalisation du modèle VGG16

```java
public class CustomVGG16ActivityModel extends VGG16ActivityModel {
    public CustomVGG16ActivityModel(Properties config) {
        super(config);
    }
    
    @Override
    protected ComputationGraph buildModel() {
        // Récupération du modèle VGG16 de base
        ComputationGraph baseModel = super.buildModel();
        
        // Modifications personnalisées
        // Par exemple, ajouter une couche supplémentaire, modifier les activations, etc.
        
        return baseModel;
    }
}
```

## Extension pour de nouveaux types de modèles

### Création d'un nouveau modèle de détection

Pour ajouter un nouveau type de modèle (par exemple, un nouveau modèle de détection d'activité), vous devez :

1. Créer une nouvelle classe qui hérite de la classe de base appropriée
2. Implémenter les méthodes requises
3. Enregistrer le nouveau modèle dans `ModelManager`

Exemple :

```java
// 1. Créer une nouvelle classe de modèle
public class EfficientNetActivityModel extends BaseActivityModel {
    public EfficientNetActivityModel(Properties config) {
        super(config);
    }
    
    @Override
    protected ComputationGraph buildModel() {
        // Construire un modèle basé sur EfficientNet
        // ...
        return model;
    }
    
    @Override
    public void loadPretrainedModel() {
        // Charger les poids pré-entraînés
        // ...
    }
    
    // Autres méthodes personnalisées
}

// 2. Ajouter le nouveau type dans ModelManager
public class ModelManager {
    // ...
    public enum ActivityModelType {
        STANDARD, VGG16, RESNET, EFFICIENTNET
    }
    
    // ...
    public BaseActivityModel getActivityModel() {
        switch (activityModelType) {
            case STANDARD:
                return new ActivityModel(config);
            case VGG16:
                return new VGG16ActivityModel(config);
            case RESNET:
                return new ResNetActivityModel(config);
            case EFFICIENTNET:
                return new EfficientNetActivityModel(config);
            default:
                return new ActivityModel(config);
        }
    }
}
```

## Personnalisation du traitement des données

### Prétraitement personnalisé des images

Pour personnaliser le prétraitement des images, vous pouvez étendre la classe `DataProcessor` ou créer une nouvelle classe utilitaire :

```java
public class CustomImagePreprocessor {
    public static INDArray preprocessImage(BufferedImage image, int height, int width) {
        // Redimensionnement
        BufferedImage resized = ImageUtils.resizeImage(image, width, height);
        
        // Prétraitements personnalisés
        // Par exemple : correction de couleur, normalisation personnalisée, etc.
        
        // Conversion en INDArray pour Deeplearning4j
        INDArray result = ImageUtils.imageToINDArray(resized);
        
        // Normalisation personnalisée
        result = result.div(255.0).sub(0.5).mul(2.0);
        
        return result;
    }
}
```

### Augmentation de données personnalisée

L'augmentation de données est une technique puissante pour améliorer les performances des modèles. Vous pouvez créer des transformations personnalisées :

```java
public class CustomDataAugmentation {
    public static ImageTransform createCustomAugmentation() {
        ImageTransform transform = new PipelineImageTransform.Builder()
            .addTransform(new RotateImageTransform(30))
            .addTransform(new FlipImageTransform(1))
            .addTransform(new WarpImageTransform(10))
            .addTransform(new ContrastNormalization(0.5))
            .build();
        
        return transform;
    }
    
    public static DataSetIterator createAugmentedIterator(DataSetIterator baseIterator) {
        ImageTransform transform = createCustomAugmentation();
        DataSetIterator augmented = new TransformingDataSetIterator(baseIterator, transform);
        return augmented;
    }
}
```

## Optimisation des performances

### Parallélisation de l'entraînement

Pour accélérer l'entraînement des modèles sur des machines multi-cœurs ou multi-GPU :

```java
// Configuration parallèle
ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
    .prefetchBuffer(8)
    .workers(4)
    .averagingFrequency(1)
    .reportScoreAfterAveraging(true)
    .build();

// Entraînement parallèle
wrapper.fit(trainDataIterator);
```

### Utilisation de la virgule flottante à mi-précision

Pour réduire la consommation de mémoire et accélérer les calculs sur les GPU compatibles :

```java
// Configuration pour utiliser la précision FP16 (half precision)
DefaultTrainingWorkspaceManager workspaceManager = WorkspaceConfiguration.builder()
    .initialSize(0)
    .overallocationLimit(0.2)
    .policyLearning(LearningPolicy.FIRST_LOOP)
    .cyclesBeforeInitialization(3)
    .policyReset(ResetPolicy.BLOCK_LEFT)
    .policySpill(SpillPolicy.EXTERNAL)
    .policyAllocation(AllocationPolicy.OVERALLOCATE)
    .build();

ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
    .weightInit(WeightInit.DISTRIBUTION)
    .activation(Activation.RELU)
    .graphBuilder()
    .addInputs("input")
    .setDataType(DataType.HALF) // Utilisation de la demi-précision (FP16)
    // ... reste de la configuration
    .build();
```

## Intégration avec d'autres outils et bibliothèques

### Export vers ONNX

Pour exporter vos modèles DL4J vers le format ONNX, utilisable dans d'autres frameworks :

```java
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorDataType;
import org.nd4j.tensorflow.conversion.TensorflowConversion;

public class OnnxExporter {
    public static void exportToOnnx(ComputationGraph model, String outputPath) {
        // Créer un prototype d'entrée pour le modèle
        INDArray inputProto = Nd4j.zeros(1, 3, 224, 224);
        
        // Exporter le modèle
        TensorflowConversion.graphToOnnxFile(model, inputProto, outputPath);
        
        System.out.println("Model exported to ONNX format at: " + outputPath);
    }
}
```

### Intégration avec Apache Spark pour l'entraînement distribué

Pour les très grands jeux de données, vous pouvez intégrer DL4J avec Spark :

```java
public class SparkTrainer {
    public static void trainModelWithSpark(ComputationGraph model, JavaSparkContext sc, String dataDir) {
        // Configuration Spark
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, 4);
        
        // Création de l'itérateur de données
        JavaRDD<DataSet> rddData = sc.textFile(dataDir)
            .map(line -> parseLineToDataSet(line));
        
        // Entraînement du modèle
        sparkNet.fit(rddData);
    }
    
    private static DataSet parseLineToDataSet(String line) {
        // Conversion d'une ligne de texte en DataSet
        // Implémentation spécifique à votre format de données
        // ...
        return dataSet;
    }
}
```

## Prochaines étapes recommandées

Pour continuer à améliorer le projet, voici quelques pistes à explorer :

1. **Implémentation de courbes ROC et PR** pour une analyse plus approfondie des performances des modèles
2. **Métriques spécialisées pour la détection d'objets** comme le mAP (mean Average Precision) et l'IoU (Intersection over Union)
3. **Exploration d'autres architectures** comme EfficientNet, DenseNet, ou des architectures transformers pour améliorer les performances
4. **Optimisation pour le déploiement embarqué** pour exécuter les modèles sur des appareils à ressources limitées
5. **API REST pour l'inférence** permettant d'utiliser les modèles via des requêtes HTTP
6. **Interface graphique de visualisation** pour interagir avec les modèles et visualiser les résultats
7. **Intégration de flux vidéo** pour la détection d'activité en temps réel
8. **Apprentissage incrémental** pour permettre aux modèles de continuer à apprendre à partir de nouvelles données
9. **Exportation vers davantage de formats** comme TensorRT, CoreML ou TensorFlow Lite pour le déploiement sur diverses plateformes
