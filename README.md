# DL4J Detection Models

Projet Java pour les modèles de détection (présence, activité et sons) utilisant DL4J.

## Description

Ce projet permet de créer et d'entraîner des modèles de détection d'activité via des images et des fichiers audio (transformés en spectrogrammes).

## Compatibilité

Ce projet a été migré vers DL4J 1.0.0-beta7.

## Modifications apportées pour la compatibilité DL4J 1.0.0-beta7

Plusieurs corrections ont été apportées pour rendre le code compatible avec DL4J 1.0.0-beta7 :

1. **Accès aux méthodes** :
   - Changement de visibilité de la méthode `getModel()` dans `MFCCSoundTrainer` et `SpectrogramSoundTrainer` de 'protected' à 'public' pour permettre l'accès depuis les tests unitaires.

2. **Compatibilité de l'API** :
   - Remplacement de la méthode `getLayerInputShape(int)` qui n'existe plus dans DL4J 1.0.0-beta7 par une approche plus robuste utilisant la méthode `getParam("W")` pour obtenir les informations sur la forme d'entrée des couches :
     - Pour les couches denses : utilisation de `model.getLayer(0).getParam("W").columns()` pour obtenir le nombre de colonnes de la matrice de poids
     - Pour les couches convolutives : utilisation de `model.getLayer(0).getParam("W").shape()[1]` pour obtenir le nombre de canaux d'entrée

3. **Adaptation des tests** :
   - Modification des tests pour utiliser les nouvelles méthodes d'accès aux formes d'entrée des couches
   - Dans `SpectrogramSoundModelTester.java`, simplification de la vérification pour se concentrer sur le nombre de canaux d'entrée et le type de couche

Ces modifications permettent une compilation réussie des tests unitaires et sont compatibles avec la version beta7 de DL4J.