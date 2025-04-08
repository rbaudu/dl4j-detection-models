#!/bin/bash

# Script pour exporter tous les modèles au format DL4J

# Définir le répertoire du projet
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_DIR"

echo "=== Exportation de tous les modèles au format DL4J ==="

# Vérifier si le projet a été compilé
if [ ! -f "target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar" ]; then
    echo "Le projet n'a pas encore été compilé. Compilation en cours..."
    ./scripts/build.sh
    
    if [ $? -ne 0 ]; then
        echo "Erreur lors de la compilation du projet. Arrêt de l'exportation."
        exit 1
    fi
fi

# Vérifier si les modèles existent
if [ ! -f "models/presence/presence_model.zip" ] || [ ! -f "models/activity/activity_model.zip" ] || [ ! -f "models/sound/sound_model.zip" ]; then
    echo "Certains modèles n'ont pas encore été entraînés. Les modèles manquants seront initialisés avec des poids aléatoires."
    echo "Il est recommandé d'entraîner les modèles avant de les exporter."
    echo
fi

# Créer le répertoire d'exportation s'il n'existe pas
mkdir -p export

# Exporter le modèle de détection de présence
echo
echo "=== Exportation du modèle de détection de présence ==="
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-presence

# Vérifier si l'exportation a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'exportation du modèle de détection de présence."
    exit 1
fi

# Exporter le modèle de détection d'activité
echo
echo "=== Exportation du modèle de détection d'activité ==="
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-activity

# Vérifier si l'exportation a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'exportation du modèle de détection d'activité."
    exit 1
fi

# Exporter le modèle de détection de sons
echo
echo "=== Exportation du modèle de détection de sons ==="
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-sound

# Vérifier si l'exportation a réussi
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'exportation du modèle de détection de sons."
    exit 1
fi

echo
echo "=== Tous les modèles ont été exportés avec succès ==="
echo "Les modèles exportés sont disponibles dans le répertoire 'export':"
echo "- export/presence_model.zip"
echo "- export/activity_model.zip"
echo "- export/sound_model.zip"
echo
echo "Ces fichiers ZIP peuvent être chargés dans d'autres applications utilisant DL4J."
