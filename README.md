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

## Documentation

La documentation complète du projet est organisée dans les fichiers suivants:

- [Installation et prérequis](docs/INSTALLATION.md) - Comment installer et configurer le projet
- [Description des modèles](docs/MODELS.md) - Détails sur les modèles implémentés
- [Guide d'utilisation](docs/USAGE.md) - Instructions d'utilisation générale
- [Système de métriques](docs/METRICS.md) - Explication du système de métriques d'évaluation
- [Export vers TensorBoard](docs/TENSORBOARD.md) - Guide sur l'exportation et l'utilisation de TensorBoard
- [Classes d'activité](docs/CLASSES.md) - Liste des 27 classes d'activité supportées
- [Utilisation avancée](docs/ADVANCED.md) - Personnalisation et utilisation avancée

## Transfert d'apprentissage

Cette implémentation utilise le **transfert d'apprentissage** avec des modèles pré-entraînés pour améliorer les performances :

- **YOLO** pour la détection de présence (détection d'objets)
- **VGG16/ResNet50** pour la détection d'activité (classification d'images)
- **YAMNet** pour la détection de sons (classification audio)
- **VGG16/ResNet50** sur spectrogrammes pour la détection de sons basée sur l'analyse visuelle

Cette approche permet d'obtenir de bonnes performances même avec un nombre limité de données d'entraînement.

## Installation rapide

```
git clone https://github.com/rbaudu/dl4j-detection-models.git
cd dl4j-detection-models
./scripts/build.sh
```

Voir [Installation et prérequis](docs/INSTALLATION.md) pour plus de détails.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
