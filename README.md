# Projet : Réseaux à Capsules (CapsNet) sur MNIST

Ce projet implémente un réseau à capsules (CapsNet) tel que décrit par Sabour et al. (2017) pour la classification et la reconstruction des chiffres manuscrits du dataset MNIST.

## Objectifs
- Comprendre l'architecture CapsNet (capsules, dynamic routing).
- Comparer avec les réseaux de neurones convolutionnels (CNN) classiques.
- Visualiser les reconstructions d'images à partir des vecteurs de capsules.

## Structure du Projet
- `model.py` : Définition de l'architecture CapsNet, PrimaryCaps, DigitCaps et du Décodeur.
- `train.py` : Script d'entraînement et d'évaluation.
- `utils.py` : Fonctions utilitaires pour le chargement des données et la reproductibilité (seeds).
- `visualize.ipynb` : Notebook pour visualiser les prédictions et les reconstructions.
- `requirements.txt` : Liste des dépendances Python.
- `LICENSE` : Licence MIT.

## Installation
1. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
Pour lancer l'entraînement :
```bash
python train.py --epochs 10 --batch-size 128 --routing 3
```

## Résultats attendus
Le modèle atteint généralement une précision > 99% sur MNIST après quelques époques. Les reconstructions permettent de vérifier que les capsules capturent bien les propriétés spatiales des chiffres.

## Reproductibilité
La seed aléatoire est fixée par défaut à 42 via `utils.set_seed()`.
