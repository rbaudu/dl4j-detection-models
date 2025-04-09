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

Les modèles sont exportés au format ZIP compatible avec DL4J, ce qui permet de les intégrer facilement dans d'autres applications Java.

## Transfert d'apprentissage

Cette implémentation utilise le **transfert d'apprentissage** avec des modèles pré-entraînés pour améliorer les performances :

- **YOLO** pour la détection de présence (détection d'objets)
- **VGG16/ResNet50** pour la détection d'activité (classification d'images)
- **YAMNet** pour la détection de sons (classification audio)

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
│   │   │       │   └── sound/       # Modèle de son (YAMNet)
│   │   │       │
│   │   │       ├── training/        # Entraînement des modèles
│   │   │       ├── export/          # Exportation des modèles
│   │   │       ├── test/            # Tests des modèles
│   │   │       ├── examples/        # Exemples d'utilisation
│   │   │       │   └── ModelUsageExample.java  # Démonstration des modèles
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
│   │   └── sound/
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
│       └── checkpoints/
│
├── export/                          # Modèles exportés pour DL4J
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

## Classes de sons

Le modèle de détection de sons prend en charge les classes suivantes par défaut :

1. Silence
2. Parole
3. Musique
4. Bruit ambiant
5. Alarme

Ces classes peuvent être personnalisées dans le fichier de configuration.

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

## Prérequis

- Java JDK 11 ou supérieur
- Maven 3.6.3 ou supérieur
- Au moins 4 Go de RAM disponible pour l'entraînement des modèles
- Au moins 8 Go de RAM recommandé pour les modèles VGG16 et ResNet

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

### Paramètres importants pour les nouveaux modèles

```properties
# Type de modèle à utiliser pour chaque tâche
presence.model.type=YOLO
activity.model.type=VGG16

# Configuration du modèle YOLO
presence.model.use.tiny.yolo=true
presence.model.input.height=416
presence.model.input.width=416

# Configuration de VGG16
activity.model.learning.rate=0.0001
activity.model.dropout=0.5
```

## Utilisation

### Sélection des modèles

Pour choisir le modèle à utiliser, vous pouvez:

1. Modifier la configuration dans `application.properties`:
   ```
   presence.model.type=YOLO
   activity.model.type=VGG16
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
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-activity-resnet
```

## Exemple d'utilisation des modèles

Voici un exemple simplifié d'utilisation des modèles dans votre code:

```java
// Charger la configuration
ConfigurationManager configManager = new ConfigurationManager();
Properties config = configManager.loadConfiguration("config/application.properties");

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

// Utiliser ResNet pour la détection d'activité
modelManager.setActivityModelType(ModelManager.ActivityModelType.RESNET);
ResNetActivityModel resNetModel = modelManager.getResNetActivityModel();
int activityClass = resNetModel.predictActivity(imageData);
```

Pour un exemple complet, voir la classe `ModelUsageExample.java`.

## Avantages des nouveaux modèles

- **YOLO** offre une meilleure localisation spatiale et une détection plus précise des personnes
- **VGG16** possède une excellente capacité de généralisation pour la classification d'images complexes
- **ResNet** utilise des connexions résiduelles pour entraîner des réseaux très profonds sans dégradation

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
