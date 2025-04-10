# Installation et Prérequis

## Prérequis

- Java JDK 11 ou supérieur
- Maven 3.6.3 ou supérieur
- Au moins 4 Go de RAM disponible pour l'entraînement des modèles
- Au moins 8 Go de RAM recommandé pour les modèles VGG16 et ResNet
- FFmpeg pour le traitement audio (pour le modèle de sons basé sur spectrogrammes)

## Installation

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
│   │   │       │       ├── AudioUtils.java              # Traitement audio
│   │   │       │       ├── DataProcessor.java           # Traitement de données
│   │   │       │       ├── EvaluationMetrics.java       # Stockage des métriques
│   │   │       │       ├── ImageUtils.java              # Traitement d'images
│   │   │       │       ├── MetricsTracker.java          # Suivi des métriques
│   │   │       │       ├── MetricsUtils.java            # Méthodes utilitaires pour les métriques
│   │   │       │       ├── MetricsVisualizer.java       # Visualisation des métriques
│   │   │       │       ├── ModelEvaluator.java          # Évaluation détaillée des modèles
│   │   │       │       ├── ModelUtils.java              # Gestion des modèles
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
│   │   │       │   ├── MetricsExampleUsage.java    # Démonstration des métriques
│   │   │       │   ├── ModelUsageExample.java      # Démonstration des modèles
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
│               ├── common/          # Tests des utilitaires communs
│               │   └── utils/       # Tests des classes utilitaires
│               │       ├── EvaluationMetricsTest.java  # Tests pour EvaluationMetrics
│               │       ├── MetricsTrackerTest.java     # Tests pour MetricsTracker
│               │       ├── MetricsUtilsTest.java       # Tests pour MetricsUtils
│               │       └── ModelEvaluatorTest.java     # Tests pour ModelEvaluator
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
│   ├── metrics/                     # Rapports et visualisations des métriques
│   │   ├── charts/                  # Graphiques des métriques
│   │   ├── reports/                 # Rapports d'évaluation
│   │   └── csv/                     # Données des métriques au format CSV
│   ├── tensorboard/                 # Logs pour TensorBoard
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

# Configuration des métriques d'évaluation
metrics.output.dir=output/metrics
test.min.accuracy=0.8
test.min.precision=0.75
test.min.recall=0.75
test.min.f1=0.75
evaluation.batch.size=32

# Configuration TensorBoard
tensorboard.enabled=true
tensorboard.log.dir=output/tensorboard
tensorboard.port=6006
tensorboard.export.epoch.frequency=1
```

Pour plus d'informations sur l'utilisation de TensorBoard, consultez le [guide TensorBoard](TENSORBOARD.md).
