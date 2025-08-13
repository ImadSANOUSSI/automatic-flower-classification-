# 🌸 Automatic Flower Classification (Streamlit)

A modern Streamlit application for automatic flower classification with a simple OpenCV-based classifier and an optional advanced ML/AI classifier. The UI and labels are localized in French.

## ✨ Features

- Classifies 5 flower species with confidence scores
- Streamlit interface only (no Flask)
- Optional advanced classifier with richer description output
- French UI and labels

## 🧱 Project Structure

```
automatic-flower-classification/
├── streamlit_app.py        # Streamlit interface
├── simple_classifier.py    # OpenCV-based baseline classifier
├── advanced_classifier.py  # Advanced ML/AI classifier
├── config.py               # Paths, classes, app configs
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── data/                   # (optional) dataset folder
├── models/                 # (optional) trained models
├── uploads/                # temp uploads
├── results/                # saved results
└── logs/                   # logs
```

## 🛠️ Requirements

- Python 3.10+
- See `requirements.txt` for full list (TensorFlow, OpenCV, Torch, Transformers, FAISS, etc.)

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: Some packages (TensorFlow, Torch, FAISS) may require specific platform wheels. If installation fails, install them individually per your OS/GPU.

## 🚀 Run Locally

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## 📥 Datasets (Kaggle)

This project can train an advanced model using these datasets:

- l3llff/flowers
- flower-photos by the TensorFlow team

Use kagglehub to download them automatically:

```bash
python scripts/download_datasets.py
```

Note: kagglehub may require you to be signed in to Kaggle in your environment.

## 🧠 Train the Advanced Model (EfficientNet)

Train a small EfficientNetB0 on the combined datasets and export to `models/`:

```bash
python scripts/train_efficientnet.py
```

This produces:

- `models/flower_efficientnet.keras`
- `models/flower_labels.json`

Once present, the app's advanced classifier will automatically use this model for inference. If absent, it falls back to the classical feature pipeline.

<!-- Flask REST API section removed: Streamlit-only project -->

## ⚙️ Configuration

Adjust settings in `config.py`:

- `FLOWER_CLASSES` / `FLOWER_CLASSES_FR`
- Model params in `CNN_CONFIG` (if used by advanced classifier)

## 📦 Models

- `simple_classifier.py`: OpenCV-based heuristic approach
- `advanced_classifier.py`: ML/AI pipeline (can leverage TensorFlow/Torch)

If no model is available/import fails, the apps gracefully fall back to mock classification for demo purposes.

## 🧹 Housekeeping

Empty folders like `data/` and `models/` are optional. Feel free to remove them if unused.

## 📄 License

MIT License. See `LICENSE`.
