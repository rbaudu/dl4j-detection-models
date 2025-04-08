# Projet de modèles de détection DL4J

Ce projet fournit une infrastructure Java complète pour l'entraînement et l'exportation de modèles de détection utilisant Deeplearning4j (DL4J). Le projet inclut trois types de modèles de détection :

1. **Détection de présence** : Détecte la présence d'objets ou de personnes
2. **Détection d'activité** : Classifie différents types d'activités (27 classes différentes)
3. **Détection de sons** : Identifie et classifie différents types de sons

Les modèles sont exportés au format ZIP compatible avec DL4J, ce qui permet de les intégrer facilement dans d'autres applications Java.

## Transfert d'apprentissage

Cette implémentation utilise le **transfert d'apprentissage** avec des modèles pré-entraînés pour améliorer les performances :

- **MobileNetV2** pour la détection d'activité (classification d'images)
- **YAMNet** pour la détection de sons (classification audio)

Cette approche permet d'obtenir de bonnes performances même avec un nombre limité de données d'entraînement.

## Structure du projet

```
project-root/
│
├── config/                          # Configuration centralisée
│   └── application.properties       # Paramètres pour tous les modèles
│
├── src/
│   ├── main/
│   │   ├── java/                    # Code source Java
│   │   │   └── com/project/
│   │   │       ├── common/          # Code commun
│   │   │       │   ├── config/      # Gestion de la configuration
│   │   │       │   └── utils/       # Classes utilitaires
│   │   │       │       ├── AudioUtils.java      # Traitement audio
│   │   │       │       ├── DataProcessor.java   # Traitement de données
│   │   │       │       ├── ImageUtils.java      # Traitement d'images
│   │   │       │       ├── ModelUtils.java      # Gestion des modèles
│   │   │       │       └── TransferLearningHelper.java  # Transfert d'apprentissage
│   │   │       │
│   │   │       ├── models/          # Définition des modèles
│   │   │       │   ├── activity/    # Modèle d'activité (MobileNetV2)
│   │   │       │   ├── presence/    # Modèle de présence
│   │   │       │   └── sound/       # Modèle de son (YAMNet)
│   │   │       │
│   │   │       ├── training/        # Entraînement des modèles
│   │   │       ├── export/          # Exportation des modèles
│   │   │       ├── test/            # Tests des modèles
│   │   │       └── Application.java # Point d'entrée
│   │   │
│   │   └── resources/               # Ressources
│   │       ├── log4j2.xml           # Configuration de logging
│   │       └── default-config.properties
│   │
│   └── test/                        # Tests unitaires et d'intégration
│       └── java/
│           └── com/project/
│               └── test/            # Tests des modèles générés
│
├── data/                            # Données d'entraînement
│   ├── raw/                         # Données brutes
│   │   ├── presence/
│   │   ├── activity/                # Images classées par répertoires d'activités
│   │   │   ├── CLEANING/
│   │   │   ├── CONVERSING/
│   │   │   ├── COOKING/
│   │   │   └── ...                  # Autres activités
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
│   ├── test_models.sh               # Script pour tester les modèles
│   └── export_all.sh
│
└── pom.xml                          # Configuration Maven
```

## Classes d'activité

Le modèle de détection d'activité prend en charge les 27 classes suivantes :

1. CLEANING - Nettoyer - Clean
2. CONVERSING - Converser, parler - Converse, speak
3. COOKING - Préparer à manger - Cook
4. DANCING - Danser - Dance
5. EATING - Manger - Eat
6. FEEDING - Nourrir le chien/chat/oiseaux/poissons - Feed
7. GOING_TO_SLEEP - Se coucher - Go to sleep
8. KNITTING - Tricoter/coudre - Knit
9. IRONING - Repasser - Iron
10. LISTENING_MUSIC - Ecouter de la musique/radio - Listen to music/radio
11. MOVING - Se déplacer - Move
12. NEEDING_HELP - Avoir besoin d'assistance - Need_help
13. PHONING - Téléphoner - Phone
14. PLAYING - Jouer - Play
15. PLAYING_MUSIC - Jouer de la musique - Play music
16. PUTTING_AWAY - Ranger - Put
17. READING - Lire - Read
18. RECEIVING - Recevoir quelqu'un - Receive
19. SINGING - Chanter - Sing
20. SLEEPING - Dormir - Sleep
21. UNKNOWN - Autre - Other
22. USING_SCREEN - Utiliser un écran (PC, laptop, tablet, smartphone) - Use screen
23. WAITING - Ne rien faire, s'ennuyer - Wait
24. WAKING_UP - Se lever - Wake up
25. WASHING - Se laver, passer aux toilettes - Wash
26. WATCHING_TV - Regarder la télévision - Watch_TV
27. WRITING - Ecrire - Write

## Classes de sons

Le modèle de détection de sons prend en charge les classes suivantes par défaut :

1. Silence
2. Parole
3. Musique
4. Bruit ambiant
5. Alarme

Ces classes peuvent être personnalisées dans le fichier de configuration.

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

### Paramètres importants pour le transfert d'apprentissage

```properties
# Configuration pour MobileNetV2 (activité)
activity.model.learning.rate=0.0005
activity.training.examples.per.class=20

# Configuration pour YAMNet (sons)
sound.model.learning.rate=0.0001
sound.model.num.classes=5

# Configuration pour les tests
test.min.accuracy=0.7
test.num.samples=100
```

## Préparation des données

### Modèle d'activité
Placez vos images d'activités dans des sous-répertoires correspondant aux noms des classes dans `data/raw/activity/`. Par exemple :
```
data/raw/activity/COOKING/image1.jpg
data/raw/activity/COOKING/image2.jpg
data/raw/activity/READING/image1.jpg
...
```

### Modèle de sons
Placez vos fichiers audio dans `data/raw/sound/`. Le système essaiera de déterminer la classe à partir du nom du fichier ou du répertoire parent.

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

### Test des modèles

Une fonctionnalité importante du projet est la capacité de tester les modèles générés avant de les exporter. Cela permet de s'assurer que les modèles sont correctement formés et pourront être chargés sans problème par l'application externe Java DL4J.

Pour tester tous les modèles :
```
./scripts/test_models.sh
```
ou
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-all
```

Pour tester un modèle spécifique :
```
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-presence
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-activity
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-sound
```

Les tests effectuent les vérifications suivantes :
1. **Chargement** : Vérifie que le modèle peut être chargé depuis le disque
2. **Validation structurelle** : S'assure que le modèle a la structure attendue et peut traiter des entrées
3. **Test de performance** : Évalue la précision du modèle sur des données de test synthétiques

Ces tests utilisent des données générées synthétiquement qui simulent les caractéristiques attendues pour chaque type de modèle. Le seuil minimal de précision acceptable est configurable via le paramètre `test.min.accuracy` dans le fichier de configuration.

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

## Workflow complet recommandé

Pour obtenir des modèles fiables et bien testés, nous recommandons le workflow suivant :

1. **Entraînement** : Entraîner les modèles avec vos données
2. **Test** : Vérifier que les modèles fonctionnent correctement
3. **Export** : Exporter les modèles validés pour utilisation externe

Exemple de script automatisant ce processus :
```bash
#!/bin/bash

# Entraîner les modèles
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar train-all

# Tester les modèles
java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar test-all

# Si les tests réussissent, exporter les modèles
if [ $? -eq 0 ]; then
    echo "Tests réussis, exportation des modèles..."
    java -jar target/dl4j-detection-models-1.0-SNAPSHOT-jar-with-dependencies.jar export-all
    echo "Exportation terminée avec succès!"
else
    echo "Tests échoués. Les modèles nécessitent une révision."
    exit 1
fi
```

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

## Avantages du transfert d'apprentissage

- **Meilleure performance** : Utilise des caractéristiques déjà apprises sur des millions d'images ou de sons
- **Moins de données requises** : Fonctionne bien même avec peu d'exemples par classe
- **Entraînement plus rapide** : Seules les dernières couches sont entraînées, ce qui accélère considérablement le processus
- **Meilleure généralisation** : Les modèles pré-entraînés ont appris des représentations robustes qui se généralisent bien à de nouvelles données

## Personnalisation des tests

Si vous souhaitez utiliser de vraies données de test au lieu de données synthétiques, vous pouvez :

1. Ajouter des sous-répertoires `test` dans les dossiers de données (`data/raw/presence/test`, etc.)
2. Placer vos données de test dans ces répertoires
3. Modifier la configuration pour pointer vers ces répertoires :

```properties
presence.test.data.dir=data/raw/presence/test
activity.test.data.dir=data/raw/activity/test
sound.test.data.dir=data/raw/sound/test
```

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
