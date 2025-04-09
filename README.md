# Projet de modèles de détection DL4J

Ce projet fournit une infrastructure Java complète pour l'entraînement et l'exportation de modèles de détection utilisant Deeplearning4j (DL4J). Le projet inclut trois types de modèles de détection :

1. **Détection de présence** : Détecte la présence d'objets ou de personnes
   - **YOLO** : Utilise YOLO pour la détection d'objets plus précise
   - **Standard** : Modèle simple pour la détection binaire de présence
2. **Détection d'activité** : Classifie différents types d'activités (27 classes différentes)
   - **VGG16** : Utilise VGG16 pour la classification d'images avec transfert d'apprentissage
   - **ResNet** : Utilise ResNet50 pour la classification d'images avec transfert d'apprentissage
   - **Standard (MobileNetV2)** : Utilise MobileNetV2 pour la classification plus légère
3. **Détection de sons** : Identifie et classifie différents types de sons
   - **Standard (YAMNet)** : Utilise YAMNet pour la classification de sons
   - **Spectrogramme** : Convertit les sons en spectrogrammes et utilise des modèles de vision (VGG16/ResNet) pour la classification

Les modèles sont exportés au format ZIP compatible avec DL4J, ce qui permet de les intégrer facilement dans d'autres applications Java.

## Transfert d'apprentissage

Cette implémentation utilise le **transfert d'apprentissage** avec des modèles pré-entraînés pour améliorer les performances :

- **YOLO** pour la détection de présence (détection d'objets)
- **VGG16/ResNet50** pour la détection d'activité (classification d'images)
- **YAMNet** pour la détection de sons (classification audio)
- **VGG16/ResNet50** sur spectrogrammes pour la détection de sons basée sur l'analyse visuelle

Cette approche permet d'obtenir de bonnes performances même avec un nombre limité de données d'entraînement.

## Structure du projet

```
project-root/
│
├── config/                          # Configuration centralisée
│   └── application.properties       # Paramètres pour tous les modèles
│
├── src/
│   ├── main/
│   │   ├── java/                    # Code source Java
│   │   │   └── com/project/
│   │   │       ├── common/          # Code commun
│   │   │       │   ├── config/      # Gestion de la configuration
│   │   │       │   └── utils/       # Classes utilitaires
│   │   │       │       ├── AudioUtils.java      # Traitement audio
│   │   │       │       ├── DataProcessor.java   # Traitement de données
│   │   │       │       ├── ImageUtils.java      # Traitement d'images
│   │   │       │       ├── ModelUtils.java      # Gestion des modèles
│   │   │       │       └── TransferLearningHelper.java  # Transfert d'apprentissage
│   │   │       │
│   │   │       ├── models/          # Définition des modèles
│   │   │       │   ├── ModelManager.java  # Gestionnaire central des modèles
│   │   │       │   ├── activity/    # Modèles d'activité
│   │   │       │   │   ├── ActivityModel.java  # Modèle standard (MobileNetV2)
│   │   │       │   │   ├── VGG16ActivityModel.java  # Modèle VGG16
│   │   │       │   │   └── ResNetActivityModel.java  # Modèle ResNet
│   │   │       │   ├── presence/    # Modèle de présence
│   │   │       │   │   ├── PresenceModel.java  # Modèle standard
│   │   │       │   │   └── YOLOPresenceModel.java  # Modèle YOLO
│   │   │       │   └── sound/       # Modèle de son
│   │   │       │       ├── SoundModel.java  # Modèle standard (YAMNet)
│   │   │       │       └── SpectrogramSoundModel.java  # Modèle basé sur spectrogrammes
│   │   │       │
│   │   │       ├── training/        # Entraînement des modèles
│   │   │       ├── export/          # Exportation des modèles
│   │   │       ├── test/            # Tests des modèles
│   │   │       ├── examples/        # Exemples d'utilisation
│   │   │       │   ├── ModelUsageExample.java  # Démonstration des modèles
│   │   │       │   └── SpectrogramSoundExample.java  # Démonstration du modèle de sons
│   │   │       └── Application.java # Point d'entrée
│   │   │
│   │   └── resources/               # Ressources
│   │       ├── log4j2.xml           # Configuration de logging
│   │       └── default-config.properties
│   │
│   └── test/                        # Tests unitaires et d'intégration
│       └── java/
│           └── com/project/
│               └── test/            # Tests des modèles générés
│
├── data/                            # Données d'entraînement
│   ├── raw/                         # Données brutes
│   │   ├── presence/
│   │   ├── activity/                # Images classées par répertoires d'activités
│   │   │   ├── CLEANING/
│   │   │   ├── CONVERSING/
│   │   │   ├── COOKING/
│   │   │   └── ...                  # Autres activités
│   │   └── sound/                   # Sons classés par répertoires d'activités
│   │       ├── CLEANING/
│   │       ├── CONVERSING/
│   │       ├── COOKING/
│   │       └── ...                  # Autres activités
│   │
│   └── processed/                   # Données prétraitées
│
├── models/                          # Modèles entraînés
│   ├── presence/
│   │   ├── presence_model.zip       # Modèle standard
│   │   ├── yolo_model.zip           # Modèle YOLO
│   │   └── checkpoints/
│   ├── activity/
│   │   ├── activity_model.zip       # Modèle standard (MobileNetV2)
│   │   ├── vgg16_model.zip          # Modèle VGG16
│   │   ├── resnet_model.zip         # Modèle ResNet
│   │   └── checkpoints/
│   └── sound/
│       ├── sound_model.zip          # Modèle standard (YAMNet)
│       ├── spectrogram_model.zip    # Modèle basé sur spectrogrammes
│       └── checkpoints/
│
├── export/                          # Modèles exportés pour DL4J
│
├── output/                          # Sorties diverses
│   └── spectrograms/                # Spectrogrammes générés
│
├── scripts/                         # Scripts utilitaires
│   ├── build.sh
│   ├── train_all.sh
│   ├── test_models.sh               # Script pour tester les modèles
│   └── export_all.sh
│
└── pom.xml                          # Configuration Maven
```

## Classes d'activité

Le modèle de détection d'activité prend en charge les 27 classes suivantes :

1. CLEANING - Nettoyer - Clean
2. CONVERSING - Converser, parler - Converse, speak
3. COOKING - Préparer à manger - Cook
4. DANCING - Danser - Dance
5. EATING - Manger - Eat
6. FEEDING - Nourrir le chien/chat/oiseaux/poissons - Feed
7. GOING_TO_SLEEP - Se coucher - Go to sleep
8. KNITTING - Tricoter/coudre - Knit
9. IRONING - Repasser - Iron
10. LISTENING_MUSIC - Ecouter de la musique/radio - Listen to music/radio
11. MOVING - Se déplacer - Move
12. NEEDING_HELP - Avoir besoin d'assistance - Need_help
13. PHONING - Téléphoner - Phone
14. PLAYING - Jouer - Play
15. PLAYING_MUSIC - Jouer de la musique - Play music
16. PUTTING_AWAY - Ranger - Put
17. READING - Lire - Read
18. RECEIVING - Recevoir quelqu'un - Receive
19. SINGING - Chanter - Sing
20. SLEEPING - Dormir - Sleep
21. UNKNOWN - Autre - Other
22. USING_SCREEN - Utiliser un écran (PC, laptop, tablet, smartphone) - Use screen
23. WAITING - Ne rien faire, s'ennuyer - Wait
24. WAKING_UP - Se lever - Wake up
25. WASHING - Se laver, passer aux toilettes - Wash
26. WATCHING_TV - Regarder la télévision - Watch_TV
27. WRITING - Ecrire - Write

## Nouveaux modèles implémentés

### Modèle YOLO pour la détection de présence

Le modèle YOLO (You Only Look Once) est un réseau de neurones convolutifs conçu pour la détection d'objets en temps réel. Dans ce projet, nous avons intégré deux versions:

- **TinyYOLO** : Version légère et rapide, idéale pour les systèmes embarqués
- **YOLO2** : Version plus complète et précise

Ces modèles permettent de détecter la présence de personnes dans les images avec une grande précision spatiale.

### Modèles VGG16 et ResNet pour la détection d'activité

Pour la classification d'activités, nous avons implémenté deux architectures de réseaux profonds pré-entraînés:

- **VGG16** : Architecture profonde à 16 couches, excellente pour la classification d'images
- **ResNet50** : Architecture résiduelle à 50 couches offrant un bon équilibre entre profondeur et performance

Ces modèles utilisent le transfert d'apprentissage à partir de poids pré-entraînés sur ImageNet, puis sont fine-tunés sur notre jeu de données d'activités.

### Modèle de reconnaissance audio basé sur spectrogrammes

Une approche innovante a été implémentée pour la détection de sons, qui combine traitement audio et vision par ordinateur :

1. **Génération de spectrogrammes** : Convertit les fichiers audio (WAV, MP3, OGG) en spectrogrammes Mel qui représentent visuellement les caractéristiques fréquentielles et temporelles des sons
2. **Classification par modèles de vision** : Utilise soit VGG16 soit ResNet50 pour classifier ces spectrogrammes
3. **Transfert d'apprentissage** : Part de modèles pré-entraînés sur ImageNet pour une meilleure généralisation

Cette approche offre plusieurs avantages :
- Meilleure représentation des sons capturant à la fois les informations temporelles et fréquentielles
- Exploitation de la puissance des modèles de vision pré-entraînés
- Visualisation des caractéristiques audio pour une meilleure compréhension
- Support de plusieurs formats audio (WAV, MP3, OGG)

## Prérequis

- Java JDK 11 ou supérieur
- Maven 3.6.3 ou supérieur
- Au moins 4 Go de RAM disponible pour l'entraînement des modèles
- Au moins 8 Go de RAM recommandé pour les modèles VGG16 et ResNet
- FFmpeg pour le traitement audio (pour le modèle de sons basé sur spectrogrammes)

## Installation et compilation

1. Clonez le dépôt :
   ```
   git clone https://github.com/rbaudu/dl4j-detection-models.git
   cd dl4j-detection-models
   ```

2. Compilez le projet :
   ```
   ./scripts/build.sh
   ```

Ce script compilera le projet et créera les répertoires nécessaires.

## Configuration

Tous les paramètres de configuration sont centralisés dans le fichier `config/application.properties`. Vous pouvez modifier ce fichier pour ajuster les paramètres des modèles, les chemins de données, etc.

### Paramètres importants pour les modèles

```properties
# Type de modèle à utiliser pour chaque tâche
presence.model.type=YOLO
activity.model.type=VGG16
sound.model.type=SPECTROGRAM

# Configuration du modèle YOLO
presence.model.use.tiny.yolo=true
presence.model.input.height=416
presence.model.input.width=416

# Configuration de VGG16
activity.model.learning.rate=0.0001
activity.model.dropout=0.5

# Configuration du modèle de sons basé sur spectrogrammes
sound.spectrogram.height=224
sound.spectrogram.width=224
sound.model.use.vgg16=true  # true pour VGG16, false pour ResNet
sound.force.retrain=false

# Paramètres d'extraction de spectrogrammes
sound.sample.rate=44100
sound.fft.size=2048
sound.hop.size=512
sound.mel.bands=128
```

## Utilisation

### Sélection des modèles

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

### Entraînement des modèles

Pour entraîner tous les modèles :
```
./scripts/train_all.sh
```

Pour entraîner un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-presence-yolo
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-activity-vgg16
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-activity-resnet
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-sound-spectrogram
```

### Test des modèles

Pour tester tous les modèles :
```
./scripts/test_models.sh
```

Pour tester un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-presence-yolo
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-activity-vgg16
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-activity-resnet
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-sound-spectrogram
```

### Exportation des modèles

Pour exporter tous les modèles au format DL4J :
```
./scripts/export_all.sh
```

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

### Entraînement du modèle

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

Pour des exemples complets, voir les classes `ModelUsageExample.java` et `SpectrogramSoundExample.java`.

## Avantages des nouveaux modèles

- **YOLO** offre une meilleure localisation spatiale et une détection plus précise des personnes
- **VGG16** possède une excellente capacité de généralisation pour la classification d'images complexes
- **ResNet** utilise des connexions résiduelles pour entraîner des réseaux très profonds sans dégradation
- **SpectrogramSoundModel** combine traitement audio et vision par ordinateur pour une meilleure reconnaissance des sons

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
