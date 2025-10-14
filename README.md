# Dog Breed Classifier (Classification de Races de Chiens)

Ce projet implémente un classificateur de races de chiens utilisant le deep learning et le transfer learning. Il permet d'identifier automatiquement trois races de chiens (Husky, Retriever, Chihuahua) à partir d'images.

## 🎯 Objectif

Développer un modèle capable de classifier automatiquement des images de chiens selon leur race, en utilisant des techniques de deep learning et de transfer learning (VGG16).

## 📊 Dataset

Le projet s'attend à une structure de données spécifique :
```
images/
    ├── husky/
    │   ├── husky1.jpg
    │   ├── husky2.jpg
    │   └── ...
    ├── retriever/
    │   ├── retriever1.jpg
    │   ├── retriever2.jpg
    │   └── ...
    └── chihuahua/
        ├── chihuahua1.jpg
        ├── chihuahua2.jpg
        └── ...
```

## 🛠️ Installation

1. Cloner le repository :
```bash
git clone https://github.com/lordraghul/DOG_BREED_CLASSIFIER.git
cd DOG_BREED_CLASSIFIER
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📖 Utilisation

Le projet est fourni sous forme d'un notebook Jupyter : `OP_ML_PROJET6_Lyes_A.ipynb`

1. Lancer Jupyter Notebook :
```bash
jupyter notebook
```

2. Ouvrir le notebook `OP_ML_PROJET6_Lyes_A.ipynb`

3. Suivre les cellules dans l'ordre pour :
   - Préparer les données (chargement et prétraitement)
   - Extraire les features avec VGG16
   - Visualiser la distribution des données (t-SNE)
   - Évaluer la séparabilité des classes (clustering)

## 🔬 Méthodologie

1. **Prétraitement** :
   - Chargement des images
   - Encodage des labels
   - Split train/test (80/20)

2. **Feature Extraction** :
   - Utilisation de VGG16 pré-entraîné
   - Extraction des features de la couche dense

3. **Analyse** :
   - Réduction de dimension (PCA + t-SNE)
   - Clustering K-means
   - Évaluation de la séparabilité (ARI)

## 📈 Performances

- Score ARI ~0.89 sur le clustering des features
- Visualisation t-SNE montrant une bonne séparation des classes
- Matrice de confusion disponible dans le notebook

## 🔗 Structure du Projet

```
DOG_BREED_CLASSIFIER/
├── README.md
├── OP_ML_PROJET6_Lyes_A.ipynb
└── requirements.txt (à créer si nécessaire)
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer de nouvelles fonctionnalités
- Soumettre une pull request

## 📝 Améliorations possibles

- Implémentation d'un classifieur supervisé complet
- Ajout de nouvelles races de chiens
- Optimisation du pipeline de données avec tf.data
- Interface utilisateur pour le test d'images

## 📫 Contact

Pour toute question ou suggestion, n'hésitez pas à :
- Ouvrir une issue sur GitHub
- Me contacter directement via mon profil GitHub

## 📜 Licence

Ce projet est sous licence MIT.