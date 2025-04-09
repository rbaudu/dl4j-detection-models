#!/bin/bash

# Script de build pour le projet de modèles de détection DL4J

# Vérifier si Maven est installé
if ! command -v mvn &> /dev/null
then
    echo "Maven n'est pas installé. Veuillez installer Maven pour continuer."
    exit 1
fi

# Définir le répertoire du projet
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_DIR"

echo "=== Construction du projet dl4j-detection-models ==="
echo "Répertoire du projet: $PROJECT_DIR"

# Nettoyer et construire le projet
echo "Nettoyage et construction du projet..."
mvn clean package

# Vérifier si la construction a réussi
if [ $? -eq 0 ]; then
    echo "Construction réussie!"
    
    # Créer les répertoires de données et de modèles s'ils n'existent pas
    mkdir -p data/raw/presence
    mkdir -p data/raw/activity
    mkdir -p data/raw/sound
    mkdir -p models/presence/checkpoints
    mkdir -p models/activity/checkpoints
    mkdir -p models/sound/checkpoints
    mkdir -p export
    
    echo "Répertoires de données et de modèles créés"
    echo
    echo "Pour exécuter l'application, utilisez:"
    echo "java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar <commande>"
    echo
    echo "Commandes disponibles:"
    echo "  train-presence-yolo      : Entraîne le modèle de détection de présence"
    echo "  train-activity-vgg16     : Entraîne le modèle de détection d'activité basé sur VGG16"
    echo "  train-activity-resnet    : Entraîne le modèle de détection d'activité basé sur ResNet"
    echo "  train-sound-spectrogram  : Entraîne le modèle de détection de sons"
    echo "  train-all         : Entraîne tous les modèles"
    echo "  export-presence   : Exporte le modèle de détection de présence"
    echo "  export-activity   : Exporte le modèle de détection d'activité"
    echo "  export-sound      : Exporte le modèle de détection de sons"
    echo "  export-all        : Exporte tous les modèles"
else
    echo "Erreur lors de la construction du projet!"
    exit 1
fi
