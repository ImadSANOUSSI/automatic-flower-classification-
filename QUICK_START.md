# 🚀 Démarrage Rapide - Classification Automatique de Fleurs

## ⚡ **Installation et Lancement en 3 étapes**

### **1. Installation des dépendances**
```bash
pip install -r requirements.txt
```

### **2. Lancement de l'application**
```bash
python main.py
```

### **3. Ouverture dans le navigateur**
Ouvrez votre navigateur sur : `http://localhost:5000`

---

## 🌺 **Fonctionnalités principales**

- **Classification automatique** de 5 espèces de fleurs
- **Interface web intuitive** en français
- **API REST** pour l'intégration
- **Génération de descriptions** avec LLaMA
- **Recherche de similarité** avec FAISS

## 📱 **Utilisation**

1. **Téléchargez** une image de fleur (JPG, PNG, BMP)
2. **Cliquez** sur "Analyser l'image"
3. **Obtenez** la classification et la description

## 🔧 **Configuration**

Modifiez `config.py` pour ajuster :
- Modèles utilisés
- Paramètres de l'API
- Configuration des modèles

## 📁 **Structure du projet**

```
automatic-flower-classification/
├── README.md              # Documentation complète
├── requirements.txt       # Dépendances Python
├── config.py             # Configuration
├── main.py               # Application principale
├── LICENSE               # Licence MIT
├── .gitignore            # Fichiers à ignorer
├── test_project.py       # Script de test
├── src/                  # Code source
│   ├── cnn_model.py     # Modèle CNN
│   ├── faiss_search.py  # Recherche FAISS
│   ├── llama_gen.py     # Génération LLaMA
│   └── utils.py         # Utilitaires
└── QUICK_START.md        # Ce fichier
```

## 🧪 **Test du projet**

```bash
python test_project.py
```

## 🆘 **Dépannage**

### **Erreur "No module named 'cv2'"**
```bash
pip install opencv-python
```

### **Erreur "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

### **Erreur "No module named 'faiss'"**
```bash
pip install faiss-cpu
```

## 🌐 **Déploiement**

### **Local**
```bash
python main.py
```

### **Production**
```bash
export FLASK_ENV=production
python main.py
```

## 📞 **Support**

- **GitHub Issues** : [Repository](https://github.com/ImadSANOUSSI/automatic-flower-classification)
- **Portfolio** : [imadsanoussi.github.io](https://imadsanoussi.github.io)

---

## 🎯 **Prochaines étapes**

1. **Installer les dépendances**
2. **Lancer l'application**
3. **Tester avec vos images**
4. **Personnaliser selon vos besoins**

**Bon développement ! 🚀**
