# Modèle de détection de présence

Ce document décrit le modèle de détection de présence, son fonctionnement, ses paramètres et son utilisation.

## Description

Le modèle de détection de présence est un réseau de neurones multicouche (MLP) conçu pour détecter la présence d'objets ou de personnes à partir de données capteurs. Il s'agit d'un modèle de classification binaire qui retourne une probabilité de présence.

## Architecture

Le modèle utilise une architecture de réseau neuronal dense avec les caractéristiques suivantes :

- **Couche d'entrée** : Taille configurable (par défaut: 64 neurones)
- **Couches cachées** : Nombre configurable (par défaut: 2 couches de 128 neurones chacune)
- **Couche de sortie** : 2 neurones (absence/présence) avec activation softmax
- **Fonction d'activation** : ReLU pour les couches cachées
- **Fonction de perte** : Entropie croisée négative (NegativeLogLikelihood)
- **Optimiseur** : Adam (configurable: Adam, Nesterovs, RMSProp)

## Paramètres de configuration

Les paramètres du modèle peuvent être ajustés dans le fichier `config/application.properties` :

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `presence.model.input.size` | Taille de la couche d'entrée | 64 |
| `presence.model.hidden.layers` | Nombre de couches cachées | 2 |
| `presence.model.hidden.size` | Taille des couches cachées | 128 |
| `presence.model.learning.rate` | Taux d'apprentissage | 0.001 |
| `presence.model.batch.size` | Taille des batchs | 32 |
| `presence.model.epochs` | Nombre d'époques d'entraînement | 100 |

## Préparation des données

Le modèle attend des données d'entrée normalisées entre 0 et 1. Pour préparer vos données :

1. Placez vos fichiers de données brutes dans le répertoire `data/raw/presence/`
2. Les données doivent être au format CSV ou texte avec une valeur par ligne
3. Chaque exemple doit contenir exactement le nombre de caractéristiques spécifié dans `presence.model.input.size`
4. Les étiquettes doivent être 0 (absence) ou 1 (présence)

## Entraînement

Pour entraîner le modèle de détection de présence :

```bash
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-presence
```

Le modèle entraîné sera sauvegardé dans `models/presence/presence_model.zip`.

## Exportation

Pour exporter le modèle au format DL4J :

```bash
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-presence
```

Le modèle exporté sera disponible dans `export/presence_model.zip`.

## Utilisation du modèle exporté

Voici un exemple de code Java pour utiliser le modèle exporté :

```java
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Charger le modèle
MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("export/presence_model.zip"));

// Préparer les données d'entrée (exemple avec 64 caractéristiques)
double[] input = new double[64];
// ... remplir le tableau input avec vos données ...

// Convertir en INDArray
INDArray inputArray = Nd4j.create(input);

// Faire une prédiction
INDArray output = model.output(inputArray);
double[] probabilities = output.toDoubleVector();

// Interpréter les résultats
double probabilityOfAbsence = probabilities[0];
double probabilityOfPresence = probabilities[1];

System.out.println("Probabilité d'absence: " + probabilityOfAbsence);
System.out.println("Probabilité de présence: " + probabilityOfPresence);
```

## Performances

Les performances du modèle dépendent de la qualité et de la quantité des données d'entraînement. Avec des données appropriées, le modèle devrait atteindre une précision supérieure à 90%.

Pour améliorer les performances :
- Augmentez la taille du jeu de données d'entraînement
- Ajustez la complexité du modèle (nombre de couches, taille des couches)
- Expérimentez avec différents taux d'apprentissage
- Utilisez la régularisation (L2, dropout) pour éviter le surapprentissage
