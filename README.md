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

## Problèmes connus

### Exceptions pendant les tests

Lors de l'exécution des tests, vous pourriez observer des exceptions dans les logs, notamment durant l'exécution de `TensorBoardExporterTest`. Ces exceptions sont liées à la bibliothèque Netty (utilisée par Vertx et DL4J) qui tente d'accéder à des classes internes de Java :

1. `UnsupportedOperationException: Reflective setAccessible(true) disabled`
2. `IllegalAccessException: class io.netty.util.internal.PlatformDependent0$6 cannot access class jdk.internal.misc.Unsafe`

**Note importante** : Ces exceptions n'affectent pas le bon fonctionnement des tests et peuvent être ignorées. Elles sont courantes avec les versions récentes de Java (Java 9+) qui ont renforcé le système de modules et de sécurité.

### Solution optionnelle

Si vous souhaitez supprimer ces messages d'erreur, vous pouvez ajouter les options suivantes à la JVM lors de l'exécution des tests :

```
--add-opens java.base/jdk.internal.misc=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED
```

Vous pouvez configurer ces options dans votre fichier `pom.xml` en ajoutant un plugin surefire avec ces arguments.