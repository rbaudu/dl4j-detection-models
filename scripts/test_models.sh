#!/bin/bash
# scripts/test_models.sh

# Ce script permet de tester les modèles de détection

# Vérifier la présence de l'argument
if [ $# -eq 0 ]; then
    MODEL="all"
else
    MODEL=$1
fi

# Variables
JAR_FILE="target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar"

# Vérifier si le JAR existe
if [ ! -f "$JAR_FILE" ]; then
    echo "Compilation du projet..."
    mvn clean package
fi

# Exécuter le test
echo "Lancement du test pour le modèle: $MODEL"
java -cp "$JAR_FILE" com.project.test.ModelTestRunner $MODEL

# Vérifier le résultat
if [ $? -eq 0 ]; then
    echo "Tests réussis!"
    exit 0
else
    echo "Tests échoués!"
    exit 1
fi