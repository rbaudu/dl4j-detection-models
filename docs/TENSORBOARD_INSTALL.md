# Installation et configuration de TensorBoard

Ce guide explique comment installer et configurer TensorBoard pour visualiser les métriques exportées par le projet.

## Prérequis

- Python 3.6 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation de TensorBoard

### Installation standard

TensorBoard peut être installé facilement via pip :

```bash
pip install tensorboard
```

Pour une installation isolée (recommandé), utilisez un environnement virtuel :

```bash
# Création d'un environnement virtuel
python -m venv tensorboard-env

# Activation de l'environnement
# Sur Windows
tensorboard-env\Scripts\activate
# Sur Linux/Mac
source tensorboard-env/bin/activate

# Installation de TensorBoard dans l'environnement
pip install tensorboard
```

### Vérification de l'installation

Vérifiez que TensorBoard est correctement installé :

```bash
tensorboard --version
```

## Configuration pour le projet DL4J

### Configuration des propriétés

Dans votre fichier `config/application.properties`, assurez-vous que les paramètres suivants sont configurés :

```properties
# Configuration TensorBoard
tensorboard.enabled=true
tensorboard.log.dir=output/tensorboard
tensorboard.port=6006
tensorboard.export.epoch.frequency=1
```

### Création des répertoires nécessaires

Assurez-vous que le répertoire de logs existe :

```bash
mkdir -p output/tensorboard
```

## Lancement de TensorBoard

### Lancement basique

Pour lancer TensorBoard et visualiser les logs exportés par le projet :

```bash
tensorboard --logdir=output/tensorboard --port=6006
```

### Lancement avec options avancées

TensorBoard propose plusieurs options avancées :

```bash
# Lancement sur une interface réseau spécifique (pour accès distant)
tensorboard --logdir=output/tensorboard --port=6006 --host=0.0.0.0

# Suivi automatique des nouvelles données
tensorboard --logdir=output/tensorboard --port=6006 --reload_interval=5
```

### Accès à TensorBoard

Une fois lancé, TensorBoard est accessible via un navigateur web à l'adresse :
- Local : [http://localhost:6006](http://localhost:6006)
- Distant (si configuré avec --host=0.0.0.0) : http://adresse-ip-serveur:6006

## Résolution des problèmes courants

### Port déjà utilisé

Si le port 6006 est déjà utilisé, vous pouvez spécifier un autre port :

```bash
tensorboard --logdir=output/tensorboard --port=8080
```

### Logs non visibles

Si vos logs n'apparaissent pas dans TensorBoard :

1. Vérifiez que `tensorboard.enabled=true` dans votre configuration
2. Assurez-vous que le chemin des logs (`tensorboard.log.dir`) est correct
3. Vérifiez les permissions du répertoire de logs
4. Essayez de vider le cache du navigateur

## Ressources supplémentaires

- [Documentation officielle de TensorBoard](https://www.tensorflow.org/tensorboard)
- [Guide TensorBoard pour TensorFlow](https://www.tensorflow.org/tensorboard/get_started)
