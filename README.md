# DL4J Detection Models

Projet Java pour les modèles de détection (présence, activité et sons) utilisant DL4J.

## Description

Ce projet permet de créer et d'entraîner des modèles de détection d'activité via des images et des fichiers audio (transformés en spectrogrammes).

## Compatibilité

Ce projet a été migré vers DL4J 1.0.0-beta7.

## Modifications apportées pour la compatibilité DL4J 1.0.0-beta7

Plusieurs corrections ont été apportées pour rendre le code compatible avec DL4J 1.0.0-beta7 :

1. Changement de visibilité de la méthode `getModel()` dans `MFCCSoundTrainer` et `SpectrogramSoundTrainer` de 'protected' à 'public' pour permettre l'accès depuis les tests unitaires.

2. Mise à jour de la méthode `getLayerInputShape(int)` qui n'existe plus dans DL4J 1.0.0-beta7, remplacée par `layerInputSize(int)` qui retourne un `long[]` au lieu d'un `int[]`.

3. Correction des types dans tous les fichiers de test pour gérer le retour de type `long[]` au lieu de `int[]` :
   - Dans `MFCCSoundModelTester.java` : Changement du type `int expectedInputSize` en `long expectedInputSize`
   - Dans `SoundTrainerTest.java` : Changement du type `int inputSize` en `long inputSize`
   - Dans `SpectrogramSoundModelTester.java` : Ajout de casts explicites `(long)channels`, `(long)height`, `(long)width` lors de la comparaison avec les éléments du tableau `long[]`.
   
4. Amélioration de la documentation dans le code concernant l'utilisation de `layerInputSize()` pour la compatibilité avec DL4J 1.0.0-beta7.

Ces modifications permettent une compilation réussie des tests unitaires et sont compatibles avec la version beta7 de DL4J.