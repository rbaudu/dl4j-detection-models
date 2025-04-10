# Guide d'utilisation

Ce guide explique comment utiliser les différentes fonctionnalités du projet.

## Sélection des modèles

Pour choisir le modèle à utiliser, vous pouvez:

1. Modifier la configuration dans `application.properties`:
   ```
   presence.model.type=YOLO
   activity.model.type=VGG16
   sound.model.type=SPECTROGRAM
   ```

2. Utiliser le ModelManager dans votre code:
   ```java
   ModelManager modelManager = new ModelManager(config);
   
   // Changer de modèle de présence
   modelManager.setPresenceModelType(ModelManager.PresenceModelType.YOLO);
   
   // Changer de modèle d'activité
   modelManager.setActivityModelType(ModelManager.ActivityModelType.RESNET);
   ```

## Entraînement des modèles

### Entraînement de tous les modèles

Pour entraîner tous les modèles :
```
./scripts/train_all.sh
```

### Entraînement d'un modèle spécifique

Pour entraîner un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-presence-yolo
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-activity-vgg16
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-activity-resnet
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-sound-spectrogram
```

## Test des modèles

### Test de tous les modèles

Pour tester tous les modèles :
```
./scripts/test_models.sh
```

### Test d'un modèle spécifique

Pour tester un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-presence-yolo
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-activity-vgg16
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-activity-resnet
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-sound-spectrogram
```

## Exportation des modèles

### Exportation de tous les modèles

Pour exporter tous les modèles au format DL4J :
```
./scripts/export_all.sh
```

### Exportation d'un modèle spécifique

Pour exporter un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-presence-yolo
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-activity-vgg16
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-sound-spectrogram
```

## Utilisation du modèle de sons basé sur spectrogrammes

### Préparation des données audio

1. Organisez vos fichiers audio (.wav, .mp3, .ogg) dans des sous-répertoires correspondant aux classes d'activités:
   ```
   data/raw/sound/COOKING/audio1.wav
   data/raw/sound/COOKING/audio2.mp3
   data/raw/sound/CONVERSING/audio1.wav
   ...
   ```
*Note :* des fichiers de son ont été extraits et classés depuis le dataset ESC-50 https://github.com/karolpiczak/ESC-50?tab=readme-ov-file.


2. Si vous n'avez pas de structure de répertoires, l'exemple `SpectrogramSoundExample` peut en créer une pour vous:
   ```java
   // Exécutez cet exemple pour créer la structure et générer des spectrogrammes
   java -cp target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar com.project.examples.SpectrogramSoundExample
   ```

### Génération et visualisation de spectrogrammes

L'exemple `SpectrogramSoundExample` peut générer et sauvegarder des spectrogrammes pour visualisation:

```java
// Générer un spectrogramme à partir d'un fichier audio
BufferedImage spectrogram = soundModel.generateSpectrogram(audioFilePath);

// Sauvegarder le spectrogramme en PNG
String outputPath = "output/spectrograms/example_spectrogram.png";
javax.imageio.ImageIO.write(spectrogram, "PNG", new File(outputPath));
```

Les spectrogrammes sont automatiquement générés lors de l'entraînement et de la prédiction, mais la visualisation peut être utile pour comprendre comment les sons sont représentés.

### Entraînement du modèle de sons

```java
// Charger la configuration
Properties config = loadConfiguration("config/application.properties");

// Créer le modèle
SpectrogramSoundModel soundModel = new SpectrogramSoundModel(config);

// Entraîner le modèle sur le jeu de données
int epochs = 100;
int batchSize = 32;
soundModel.trainOnDataset("data/raw/sound", epochs, batchSize);

// Sauvegarder le modèle
soundModel.saveModel("models/sound/spectrogram_model.zip");
```

### Prédiction

```java
// Charger un modèle existant
soundModel.loadModel("models/sound/spectrogram_model.zip");

// Prédire la classe d'un fichier audio
String audioFilePath = "path/to/audio/file.wav";
String predictedClass = soundModel.predict(audioFilePath);

System.out.println("Classe prédite: " + predictedClass);
```

## Exemple d'utilisation des modèles

Voici un exemple simplifié d'utilisation des modèles dans votre code:

```java
// Charger la configuration
Properties config = loadConfiguration("config/application.properties");

// Créer le gestionnaire de modèles
ModelManager modelManager = new ModelManager(config);

// Utiliser YOLO pour la détection de présence
modelManager.setPresenceModelType(ModelManager.PresenceModelType.YOLO);
YOLOPresenceModel yoloModel = modelManager.getYoloPresenceModel();
boolean presenceDetected = yoloModel.detectPresence(imageData);

// Utiliser VGG16 pour la détection d'activité
modelManager.setActivityModelType(ModelManager.ActivityModelType.VGG16);
VGG16ActivityModel vgg16Model = modelManager.getVgg16ActivityModel();
int activityClass = vgg16Model.predictActivity(imageData);

// Utiliser le modèle de sons basé sur spectrogrammes
SpectrogramSoundModel soundModel = new SpectrogramSoundModel(config);
soundModel.loadModel("models/sound/spectrogram_model.zip");
String predictedSound = soundModel.predict(audioFilePath);
```

Pour des exemples complets, voir les classes `ModelUsageExample.java`, `SpectrogramSoundExample.java` et `MetricsExampleUsage.java`.

## Utilisation des métriques et visualisation

Consultez le fichier [METRICS.md](METRICS.md) pour des informations détaillées sur l'utilisation du système de métriques d'évaluation.

## TensorBoard pour la visualisation des métriques

Consultez le fichier [TENSORBOARD.md](TENSORBOARD.md) pour des informations sur l'exportation des métriques vers TensorBoard et l'utilisation de cet outil pour visualiser l'évolution de l'entraînement.
