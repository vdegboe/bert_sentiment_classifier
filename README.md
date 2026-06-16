# BERT Sentiment Classification

Classification automatique des sentiments exprimés dans des tweets en utilisant un modèle BERT fine-tuné. Le projet permet d'entraîner un modèle, d'évaluer ses performances et de faire des prédictions sur de nouveaux tweets via une interface web.

## Structure du Projet

```
bert-classification-sentiment/
├── data/                      # Dataset (téléchargé automatiquement)
├── checkpoint/                # Modèles sauvegardés après entraînement
├── dataset.py                 # Classe pour charger et tokenizer les données
├── model.py                   # Architecture du modèle BERT
├── train.py                   # Script principal d'entraînement
├── demo.py                    # Interface web Gradio pour tester le modèle
├── utils.py                   # Fonctions utilitaires (métriques, visualisations)
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

## Présentation du Dataset

### Source et caractéristiques

Le dataset utilisé provient d'une collection de tweets annotés manuellement avec des étiquettes de sentiment. Il est téléchargé automatiquement depuis Google Drive lors de la première exécution du programme (fichier `dataset.csv`).

### Classes de sentiments

Le dataset contient 5 classes de sentiments représentant différents niveaux d'expression :

- 0 : Extremely Negative - expressions très négatives
- 1 : Negative - sentiments négatifs
- 2 : Neutral - sentiments neutres
- 3 : Positive - sentiments positifs
- 4 : Extremely Positive - sentiments très positifs

### Statistiques et distribution

Le dataset contient 41 157 tweets répartis inégalement entre les 5 classes :

- Classe 0 (Extremely Negative) : 5 481 exemples (13.32%)
- Classe 1 (Negative) : 9 917 exemples (24.10%)
- Classe 2 (Neutral) : 7 713 exemples (18.74%)
- Classe 3 (Positive) : 11 422 exemples (27.75%)
- Classe 4 (Extremely Positive) : 6 624 exemples (16.09%)

La classe Positive est la plus représentée (27.75%) tandis que Extremely Negative est la moins représentée (13.32%), créant un ratio de déséquilibre d'environ 2.08:1.

Pour gérer ce déséquilibre, le modèle utilise des poids de classe (class weights) proportionnels à l'inverse de la fréquence de chaque classe. Cela pénalise les erreurs sur les classes peu représentées et évite que le modèle se contente de prédire toujours la classe la plus fréquente.

### Exemples

- Tweet très négatif : "Je déteste vraiment cette situation, c'est un désastre total"
- Tweet négatif : "Ce n'est pas très bon"
- Tweet neutre : "Les tweets peuvent parler de n'importe quoi"
- Tweet positif : "J'aime bien cette idée"
- Tweet très positif : "C'est absolument incroyable, j'adore!"

## Description du Modèle et Choix Techniques

### Architecture générale

Le modèle combine BERT (Bidirectional Encoder Representations from Transformers) avec une tête de classification simple. BERT est un modèle pré-entraîné sur une grande quantité de texte anglais, capable de comprendre les nuances du langage naturel.

### Choix du tokenizer

Le tokenizer BERT utilisé est `BertTokenizer.from_pretrained("bert-base-uncased")`. Ce tokenizer :
- Convertit le texte en minuscules (uncased)
- Divise les mots en sous-mots (tokens) pour gérer les mots rares
- Ajoute des tokens spéciaux : [CLS] au début et [SEP] à la fin
- Génère un masque d'attention pour ignorer les tokens de padding

### Longueur maximale des séquences

La longueur maximale des tweets est fixée à 128 tokens. Ce choix résulte d'un équilibre entre :
- Les tweets complets (rarement plus de 128 tokens après tokenization)
- La mémoire GPU disponible
- Le temps de calcul

Les tweets plus longs sont tronqués, les plus courts sont complétés avec du padding.

### Architecture de la tête de classification

```
BERT (base-uncased) - 768 dimensions
    |
Dropout (30%)
    |
Couche linéaire (768 -> 256 neurones)
    |
ReLU (activation)
    |
Dropout (30%)
    |
Couche linéaire (256 -> 5 classes)
    |
Softmax (probabilités)
```

La première couche réduit les 768 dimensions de BERT à 256. Le dropout régularise le modèle pour éviter le surapprentissage. La seconde couche produit 5 scores (un par classe).

### Hyperparamètres d'entraînement

- Batch Size : 32 (nombre d'exemples traités en parallèle)
- Learning Rate : 2e-5 (petit taux d'apprentissage pour ne pas déstabiliser BERT)
- Epochs : 5 (nombre de passages sur le dataset)
- Early Stopping : arrêt si la validation ne s'améliore pas pendant 3 epochs
- Warmup : augmentation progressive du learning rate les 10% premiers steps
- Loss : CrossEntropyLoss avec class weights pour gérer le déséquilibre

## Étapes de Réalisation

### Phase 1 : Préparation des données
1. Télécharger le dataset depuis Google Drive
2. Charger et nettoyer le CSV
3. Mapper les étiquettes textuelles aux indices numériques
4. Diviser en ensembles train (70%), validation (14%), test (16%)

### Phase 2 : Construction du modèle
1. Charger le modèle BERT pré-entraîné
2. Ajouter une tête de classification (couches linéaires)
3. Configurer l'optimiseur (AdamW) et le scheduler de learning rate
4. Mettre en place la sauvegarde du meilleur modèle

### Phase 3 : Entraînement
1. Itérer sur les epochs
2. Pour chaque batch : forward pass, calcul de loss, backward pass, optimisation
3. Valider après chaque epoch
4. Appliquer l'early stopping si nécessaire
5. Sauvegarder le modèle avec les meilleures performances

### Phase 4 : Évaluation et visualisations
1. Calculer les métriques sur l'ensemble test (accuracy, F1-score, matrice de confusion)
2. Générer des graphiques pour visualiser la convergence
3. Analyser les erreurs de classification

### Phase 5 : Interface de démonstration
1. Créer une interface web avec Gradio
2. Charger le modèle entraîné
3. Permettre à l'utilisateur d'entrer du texte et obtenir les prédictions

## Difficultés Rencontrées

### 1. Temps d'entraînement très long

L'entraînement complet du modèle a pris plusieurs heure, même avec un GPU sur Google Colab. Cette lenteur a été la principale difficulté pratique rencontrée.


Solutions appliquées :
- Utilisation de Google Colab pour bénéficier d'une GPU gratuite
- Réduction du nombre d'epochs à 5

### 2. Erreurs de type avec le tokenizer BERT

Durant le développement, des bugs  ont surgi liés au format des données retournées par le tokenizer. Le tokenizer retourne des tenseurs avec une dimension batch (taille 1) automatiquement ajoutée quand on lui passe un seul exemple, ce qui causait des incompatibilités avec le reste du pipeline.


Solutions appliquées :
- Ajout de `.squeeze()` dans la méthode `__getitem__` de CustomDataset pour supprimer la dimension batch automatique
- Vérification  des shapes des tenseurs 


### 3. Déséquilibre  entre les classes

Le dataset avait une distribution clairement déséquilibrée : la classe Positive représentait 27.75% des données, Negative 24.10%, tandis que Extremely Negative n'en représentait que 13.32%. Ce déséquilibre de 2.08:1 entre la classe la plus fréquente et la moins fréquente aurait pu causer un biais dans les prédictions du modèle.


Solutions appliquées :
- Calcul automatique des class weights inversement proportionnels à la fréquence de chaque classe
- Intégration des poids dans la loss function (CrossEntropyLoss avec weight parameter)
- Utilisation du F1-score weighted pour évaluer le modèle de manière équitable

### 4. Indexation des classes hors limites (out of bounds)

Au début du projet, les classes étaient encodées de 1 à 5 (Extremely Negative = 1, Extremely Positive = 5). Lors du premier entraînement, le modèle levait une erreur car il attendait des indices de classe entre 0 et 4 (5 classes, indexation Python standard).


Solutions appliquées :
- Modification du mapping des sentiments dans dataset.py pour passer de (1-5) à (0-4)


### 5. Problème d'indexation du DataFrame lors du split train/test

Après le split du dataset en train/validation/test avec `train_test_split`, les indices du DataFrame n'étaient pas continus (par exemple : [0, 2, 5, 8, ...] au lieu de [0, 1, 2, 3, ...]). Cela causait des erreurs d'accès aux données dans le DataLoader.


Solutions appliquées :
- Utilisation de `reset_index(drop=True)` sur chaque sous-ensemble après le split pour réinitialiser les indices à [0, 1, 2, ...]
- Application de `.reset_index(drop=True)` dans le constructeur de CustomDataset pour assurer que les indices correspondent toujours aux positions


## Installation

Avant de commencer, assurez-vous d'avoir Python 3.8+ installé.

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Les dépendances principales incluent :
- torch : framework deep learning
- transformers : modèles pré-entraînés BERT
- pandas : manipulation de données
- scikit-learn : métriques et séparation train/test
- matplotlib : visualisations
- gradio : interface web
- gdown : téléchargement depuis Google Drive

## Exécution

### Entraînement du modèle

```bash
python train.py
```

Ce script va :
1. Télécharger automatiquement le dataset (première exécution seulement)
2. Afficher les statistiques du dataset
3. Diviser les données en train/validation/test
4. Entraîner le modèle BERT pendant 5 epochs maximum (avec early stopping)
5. Sauvegarder le meilleur modèle dans le dossier `checkpoint/`
6. Générer des graphiques (convergence, matrice de confusion) dans le dossier `outputs/`

Durée estimée : 30-60 minutes sans GPU, 10-20 minutes avec GPU.

### Utilisation de la démo interactive

Une fois l'entraînement terminé, lancez l'interface web :

```bash
python demo.py
```

L'interface s'ouvrira automatiquement dans votre navigateur. Vous pouvez alors :
1. Écrire ou coller un tweet
2. Cliquer sur "Submit" pour obtenir la prédiction
3. Voir les probabilités pour chaque classe de sentiment

## Résultats et Métriques

### Résultats de l'Entraînement

Le modèle a été entraîné pendant 5 epochs. Voici l'évolution des performances :

| Epoch | Train Loss | Train Acc | Eval Loss | Eval Acc | Eval F1 | Status |
|-------|-----------|-----------|-----------|----------|---------|--------|
| 1     | 0.8347    | 0.6442    | 0.5314    | 0.7761   | 0.7738  | ✓ Meilleur |
| 2     | 0.4341    | 0.8354    | 0.4155    | 0.8339   | 0.8337  | ✓ Meilleur |
| 3     | 0.2955    | 0.8899    | 0.3700    | 0.8728   | 0.8729  | ✓✓ Meilleur |
| 4     | 0.2168    | 0.9220    | 0.3913    | 0.8684   | 0.8684  |  |
| 5     | 0.1641    | 0.9416    | 0.4448    | 0.8556   | 0.8551  |  |

### Meilleur Modèle (Epoch 3)

Le meilleur modèle a été sauvegardé à l'Epoch 3 avec les performances suivantes :

**Résultats de Validation (Epoch 3)**
- Loss : 0.3700
- Accuracy : 0.8728 (87.28%)
- F1-Score (weighted) : 0.8729

**Résultats sur l'Ensemble de Test**
- Loss : 0.4448
- Accuracy : 0.8556 (85.56%)
- F1-Score (weighted) : 0.8551

### Observations Clés

- Le modèle converge rapidement (amélioration majeure entre l'Epoch 1 et 3)
- L'Epoch 3 représente le meilleur équilibre entre performance de validation et généralisation
- Après l'Epoch 3, le modèle commence à surapprendre légèrement (validation loss augmente)
- La performance test (85.56% accuracy) valide que le modèle généralise bien sur des données inédites

### Métriques Sauvegardées

Le modèle entraîné sauvegarde les métriques suivantes :

- **Accuracy** : pourcentage de prédictions correctes
- **F1-Score (weighted)** : moyenne pondérée du F1-score pour chaque classe
- **Matrice de confusion** : montre les erreurs de classification
- **Courbes de convergence** : visualisent loss et accuracy à travers les epochs

Ces résultats sont affichés dans la console et sauvegardés sous forme de graphiques dans le dossier `outputs/`.

## Fichiers Principaux

- `dataset.py` : Classe CustomDataset pour charger et tokenizer les tweets
- `model.py` : Classe MyBertModel définissant l'architecture du classifier
- `train.py` : Script principal contenant la boucle d'entraînement
- `utils.py` : Fonctions d'entraînement, évaluation et génération de graphiques
- `demo.py` : Interface Gradio pour l'inférence interactive

## Notes Techniques

- Le modèle supporte automatiquement GPU (CUDA) et CPU
- Les tokens spéciaux [CLS] et [SEP] sont gérés automatiquement par le tokenizer
- La classe des sentiments est mappée numériquement pour l'entraînement
- Le modèle utilise la détection de device automatique (device = "cuda" si disponible sinon "cpu")
