# 🌸 Automatic Flower Classification

**Deep Learning model for automatic flower classification using CNN, FAISS, and LLaMA**

## 📋 Description

Ce projet implémente un système de classification automatique de fleurs utilisant des techniques avancées de Deep Learning. Le système combine un CNN (Convolutional Neural Network) pour l'extraction de caractéristiques, FAISS pour la recherche de similarité rapide, et LLaMA pour la génération de descriptions naturelles.

## 🚀 Fonctionnalités

- **Classification automatique** de 5 espèces de fleurs
- **Extraction de caractéristiques** avec CNN pré-entraîné
- **Recherche de similarité** avec FAISS
- **Génération de descriptions** avec LLaMA
- **Interface utilisateur** intuitive
- **API REST** pour l'intégration

## 🌺 Espèces supportées

1. **Daisy** (Marguerite)
2. **Dandelion** (Pissenlit)
3. **Rose** (Rose)
4. **Sunflower** (Tournesol)
5. **Tulip** (Tulipe)

## 🛠️ Technologies utilisées

- **Python 3.8+**
- **TensorFlow/Keras** - Deep Learning
- **FAISS** - Recherche de similarité
- **LLaMA** - Génération de texte
- **Flask** - API web
- **OpenCV** - Traitement d'images
- **NumPy/Pandas** - Manipulation de données

## 📦 Installation

### Prérequis
```bash
Python 3.8+
pip
```

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Configuration
```bash
python config.py
```

### 2. Lancement
```bash
python main.py
```

### 3. Interface web
Ouvrez votre navigateur sur `http://localhost:5000`

## 📁 Structure du projet

```
automatic-flower-classification/
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances Python
├── config.py             # Configuration du projet
├── main.py               # Script principal
├── LICENSE               # Licence MIT
├── models/               # Modèles pré-entraînés
├── data/                 # Données d'entraînement
├── src/                  # Code source
│   ├── cnn_model.py     # Modèle CNN
│   ├── faiss_search.py  # Recherche FAISS
│   ├── llama_gen.py     # Génération LLaMA
│   └── utils.py         # Utilitaires
└── tests/                # Tests unitaires
```

## 🔧 Configuration

Modifiez `config.py` pour ajuster :
- Chemins des modèles
- Paramètres du CNN
- Configuration FAISS
- Paramètres LLaMA

## 📊 Performance

- **Précision** : 95%+ sur le jeu de données de test
- **Temps de réponse** : < 2 secondes
- **Support** : 5 espèces de fleurs

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :
1. Fork le projet
2. Créer une branche feature
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

**Imad SANOUSSI**
- GitHub: [@ImadSANOUSSI](https://github.com/ImadSANOUSSI)
- Portfolio: [Portfolio Web](https://imadsanoussi.github.io)

## 🙏 Remerciements

- Dataset: [Flower Classification Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- Modèles pré-entraînés: TensorFlow Hub
- Architecture inspirée de recherches récentes en Computer Vision

## 📞 Support

Pour toute question ou problème :
- Ouvrez une issue sur GitHub
- Contactez-moi via mon portfolio

---

⭐ **N'oubliez pas de donner une étoile au projet si vous l'aimez !**
