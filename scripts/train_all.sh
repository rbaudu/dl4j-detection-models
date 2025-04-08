#!/bin/bash

# Script pour entraîner tous les modèles de détection

# Définir le répertoire du projet
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_DIR"

echo "=== Entraînement de tous les modèles de détection ==="

# Vérifier si le projet a été compilé
if [ ! -f "target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar" ]; then
    echo "Le projet n'a pas encore été compilé. Compilation en cours..."
    ./scripts/build.sh
    
    if [ $? -ne 0 ]; then
        echo "Erreur lors de la compilation du projet. Arrêt de l'entraînement."
        exit 1
    fi
fi

# Entraîner le modèle de détection de présence
echo
echo "=== Entraînement du modèle de détection de présence ==="
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-presence

# Vérifier si l'entraînement a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'entraînement du modèle de détection de présence."
    exit 1
fi

# Entraîner le modèle de détection d'activité
echo
echo "=== Entraînement du modèle de détection d'activité ==="
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-activity

# Vérifier si l'entraînement a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'entraînement du modèle de détection d'activité."
    exit 1
fi

# Entraîner le modèle de détection de sons
echo
echo "=== Entraînement du modèle de détection de sons ==="
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-sound

# Vérifier si l'entraînement a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'entraînement du modèle de détection de sons."
    exit 1
fi

echo
echo "=== Tous les modèles ont été entraînés avec succès ==="
echo "Les modèles entraînés sont disponibles dans le répertoire 'models'."
