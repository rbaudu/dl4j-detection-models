# Placeholders pour les captures d'écran TensorBoard

Ce fichier sert de rappel des captures d'écran TensorBoard à ajouter pour compléter la documentation.

## Captures d'écran à ajouter

Lors de votre prochain entraînement de modèle avec TensorBoard activé, prenez des captures d'écran des interfaces suivantes et placez-les dans ce répertoire :

1. **tensorboard_scalars.png** - Onglet "Scalars" montrant l'évolution des métriques (accuracy, loss, precision, recall)
2. **tensorboard_confusion_matrix.png** - Onglet "Images" montrant la matrice de confusion exportée
3. **tensorboard_graph.png** - Onglet "Graphs" montrant la structure du modèle
4. **tensorboard_comparison.png** - Visualisation comparant plusieurs modèles sur les mêmes données

## Mise à jour de la documentation

Une fois les captures d'écran réalisées, mettez à jour le fichier `TENSORBOARD.md` pour inclure ces images avec des références comme :

```markdown
![TensorBoard Scalars](images/tensorboard_scalars.png)
*Exemple de visualisation des métriques d'accuracy, loss, precision et recall au fil des époques*
```

Les emplacements recommandés pour ces images sont déjà indiqués dans le fichier `TENSORBOARD.md` sous forme de commentaires.

## Exemples de ce à quoi devraient ressembler les captures

- **tensorboard_scalars.png** : Graphiques d'évolution montrant accuracy, loss, precision et recall au fil des époques
- **tensorboard_confusion_matrix.png** : Matrice colorée montrant les prédictions correctes et incorrectes
- **tensorboard_graph.png** : Représentation graphique de l'architecture du réseau (par exemple VGG16)
- **tensorboard_comparison.png** : Graphiques superposés comparant les performances de différents modèles
