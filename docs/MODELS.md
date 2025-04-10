# Modèles Implémentés

Ce projet implémente plusieurs types de modèles de détection pour différentes tâches. Voici une description détaillée de chaque modèle.

## Modèles de détection de présence

### Modèle YOLO pour la détection de présence

Le modèle YOLO (You Only Look Once) est un réseau de neurones convolutifs conçu pour la détection d'objets en temps réel. Dans ce projet, nous avons intégré deux versions:

- **TinyYOLO** : Version légère et rapide, idéale pour les systèmes embarqués
- **YOLO2** : Version plus complète et précise

Ces modèles permettent de détecter la présence de personnes dans les images avec une grande précision spatiale.

### Modèle standard de détection de présence

Ce modèle plus simple est basé sur un réseau de neurones convolutifs classique optimisé pour la détection binaire (présence/absence). Il est plus léger que YOLO mais moins précis pour la localisation.

## Modèles de détection d'activité

### Modèles VGG16 et ResNet pour la détection d'activité

Pour la classification d'activités, nous avons implémenté deux architectures de réseaux profonds pré-entraînés:

- **VGG16** : Architecture profonde à 16 couches, excellente pour la classification d'images
- **ResNet50** : Architecture résiduelle à 50 couches offrant un bon équilibre entre profondeur et performance

Ces modèles utilisent le transfert d'apprentissage à partir de poids pré-entraînés sur ImageNet, puis sont fine-tunés sur notre jeu de données d'activités.

### Modèle standard de détection d'activité (MobileNetV2)

Ce modèle utilise l'architecture MobileNetV2, qui est optimisée pour les appareils mobiles et les systèmes avec des ressources limitées, tout en maintenant de bonnes performances de classification.

## Modèles de détection de sons

### Modèle standard de détection de sons (YAMNet)

Ce modèle utilise l'architecture YAMNet, qui est spécialement conçue pour la classification audio et la détection d'événements sonores.

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

## Avantages des modèles implémentés

- **YOLO** offre une meilleure localisation spatiale et une détection plus précise des personnes
- **VGG16** possède une excellente capacité de généralisation pour la classification d'images complexes
- **ResNet** utilise des connexions résiduelles pour entraîner des réseaux très profonds sans dégradation
- **SpectrogramSoundModel** combine traitement audio et vision par ordinateur pour une meilleure reconnaissance des sons
- **MobileNetV2** offre un bon compromis entre performance et ressources nécessaires

## Configuration des modèles

Pour configurer les différents modèles, vous pouvez modifier les paramètres dans le fichier `config/application.properties`. Consultez le [guide d'installation](INSTALLATION.md) pour plus de détails sur les paramètres de configuration disponibles.
