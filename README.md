# Projet de modèles de détection DL4J

Ce projet fournit une infrastructure Java complète pour l'entraînement et l'exportation de modèles de détection utilisant Deeplearning4j (DL4J). Le projet inclut trois types de modèles de détection :

1. **Détection de présence** : Détecte la présence d'objets ou de personnes
2. **Détection d'activité** : Classifie différents types d'activités
3. **Détection de sons** : Identifie et classifie différents types de sons

Les modèles sont exportés au format ZIP compatible avec DL4J, ce qui permet de les intégrer facilement dans d'autres applications Java.

## Structure du projet

```
project-root/
│
├── config/                          # Configuration centralisée
│   └── application.properties       # Paramètres pour tous les modèles
│
├── src/
│   └── main/
│       ├── java/                    # Code source Java
│       │   └── com/project/
│       │       ├── common/          # Code commun
│       │       ├── models/          # Définition des modèles
│       │       ├── training/        # Entraînement des modèles
│       │       ├── export/          # Exportation des modèles
│       │       └── Application.java # Point d'entrée
│       │
│       └── resources/               # Ressources
│           ├── log4j2.xml           # Configuration de logging
│           └── default-config.properties
│
├── data/                            # Données d'entraînement
│   ├── raw/                         # Données brutes
│   │   ├── presence/
│   │   ├── activity/
│   │   └── sound/
│   │
│   └── processed/                   # Données prétraitées
│
├── models/                          # Modèles entraînés
│   ├── presence/
│   │   └── checkpoints/
│   ├── activity/
│   │   └── checkpoints/
│   └── sound/
│       └── checkpoints/
│
├── export/                          # Modèles exportés pour DL4J
│
├── scripts/                         # Scripts utilitaires
│   ├── build.sh
│   ├── train_all.sh
│   └── export_all.sh
│
└── pom.xml                          # Configuration Maven
```

## Prérequis

- Java JDK 11 ou supérieur
- Maven 3.6.3 ou supérieur
- Au moins 4 Go de RAM disponible pour l'entraînement des modèles

## Installation et compilation

1. Clonez le dépôt :
   ```
   git clone https://github.com/rbaudu/dl4j-detection-models.git
   cd dl4j-detection-models
   ```

2. Compilez le projet :
   ```
   ./scripts/build.sh
   ```

Ce script compilera le projet et créera les répertoires nécessaires.

## Configuration

Tous les paramètres de configuration sont centralisés dans le fichier `config/application.properties`. Vous pouvez modifier ce fichier pour ajuster les paramètres des modèles, les chemins de données, etc.

## Utilisation

### Entraînement des modèles

Pour entraîner tous les modèles :
```
./scripts/train_all.sh
```

Pour entraîner un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-presence
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-activity
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-sound
```

### Exportation des modèles

Pour exporter tous les modèles au format DL4J :
```
./scripts/export_all.sh
```

Pour exporter un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-presence
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-activity
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-sound
```

Les modèles exportés seront placés dans le répertoire `export/` au format ZIP.

## Utilisation des modèles exportés

Les modèles exportés peuvent être chargés dans d'autres applications utilisant DL4J avec le code suivant :

```java
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

// Charger le modèle
MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("chemin/vers/le/modele.zip"));

// Utiliser le modèle pour faire des prédictions
INDArray input = ... // Préparer les données d'entrée
INDArray output = model.output(input);
```

## Personnalisation

### Ajout de nouvelles données d'entraînement

Pour ajouter vos propres données d'entraînement, placez-les dans les répertoires correspondants sous `data/raw/` :
- `data/raw/presence/` pour les données de détection de présence
- `data/raw/activity/` pour les données de détection d'activité
- `data/raw/sound/` pour les données de détection de sons

### Modification des paramètres des modèles

Modifiez le fichier `config/application.properties` pour ajuster les paramètres des modèles, tels que :
- Taille des couches
- Taux d'apprentissage
- Nombre d'époques
- Taille des batches
- etc.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
