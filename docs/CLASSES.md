# Classes d'activité

Le modèle de détection d'activité prend en charge les 27 classes suivantes :

| # | Anglais | Français | Description |
|---|---------|----------|-------------|
| 1 | CLEANING | Nettoyer | Activités de nettoyage, ménage |
| 2 | CONVERSING | Converser, parler | Communication verbale entre personnes |
| 3 | COOKING | Préparer à manger | Préparation de repas, cuisine |
| 4 | DANCING | Danser | Mouvements rythmiques, danse |
| 5 | EATING | Manger | Prise de repas, consommation de nourriture |
| 6 | FEEDING | Nourrir | Nourrir des animaux (chien/chat/oiseaux/poissons) |
| 7 | GOING_TO_SLEEP | Se coucher | Préparation au coucher |
| 8 | KNITTING | Tricoter/coudre | Activités de tricot, couture |
| 9 | IRONING | Repasser | Repassage de vêtements |
| 10 | LISTENING_MUSIC | Ecouter de la musique/radio | Écoute attentive de musique ou radio |
| 11 | MOVING | Se déplacer | Déplacement dans l'espace |
| 12 | NEEDING_HELP | Avoir besoin d'assistance | Situation nécessitant de l'aide |
| 13 | PHONING | Téléphoner | Communication téléphonique |
| 14 | PLAYING | Jouer | Jeux, divertissement |
| 15 | PLAYING_MUSIC | Jouer de la musique | Performance musicale avec instrument |
| 16 | PUTTING_AWAY | Ranger | Activités de rangement |
| 17 | READING | Lire | Lecture de livres, journaux, etc. |
| 18 | RECEIVING | Recevoir quelqu'un | Accueil de visiteurs |
| 19 | SINGING | Chanter | Performance vocale |
| 20 | SLEEPING | Dormir | État de sommeil |
| 21 | UNKNOWN | Autre | Activités non catégorisées |
| 22 | USING_SCREEN | Utiliser un écran | Utilisation d'appareils électroniques (PC, laptop, tablet, smartphone) |
| 23 | WAITING | Ne rien faire, s'ennuyer | Attente, inactivité |
| 24 | WAKING_UP | Se lever | Sortie du sommeil, réveil |
| 25 | WASHING | Se laver, passer aux toilettes | Hygiène personnelle |
| 26 | WATCHING_TV | Regarder la télévision | Visionnage de programmes TV |
| 27 | WRITING | Ecrire | Écriture manuelle ou sur clavier |

## Organisation des données

Pour l'entraînement des modèles de détection d'activité, les données doivent être organisées selon cette structure de dossiers :

```
data/raw/activity/
├── CLEANING/
├── CONVERSING/
├── COOKING/
├── DANCING/
├── ...
└── WRITING/
```

Chaque dossier doit contenir des images de l'activité correspondante.

Pour les modèles de sons basés sur spectrogrammes, une structure similaire est utilisée :

```
data/raw/sound/
├── CLEANING/
├── CONVERSING/
├── COOKING/
├── DANCING/
├── ...
└── WRITING/
```

Avec des fichiers audio (WAV, MP3, OGG) dans chaque dossier correspondant à l'activité.

## Détection automatique des classes

Le système prend en charge la détection automatique des classes à partir de la structure des dossiers. Ce comportement est contrôlé par les paramètres suivants dans `application.properties` :

```properties
activity.auto.detect.classes=true
sound.auto.detect.classes=true
```

Si ces options sont activées, le système parcourt les dossiers de données et crée automatiquement une liste des classes disponibles.

## Métriques par classe

Le système de métriques d'évaluation (voir [METRICS.md](METRICS.md)) permet d'analyser les performances du modèle pour chaque classe individuellement. Cela permet d'identifier les classes qui sont bien reconnues et celles qui nécessitent des améliorations.

Les rapports générés incluent des métriques détaillées pour chaque classe, telles que la précision, le rappel et le F1-score.
