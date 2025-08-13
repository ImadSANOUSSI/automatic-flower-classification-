# 🌸 Configuration file for Automatic Flower Classification
# Author: Imad SANOUSSI
# GitHub: https://github.com/ImadSANOUSSI

import os
from pathlib import Path

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Project name and version
PROJECT_NAME = "automatic-flower-classification"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Deep Learning model for automatic flower classification"

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# CNN Model Configuration
CNN_CONFIG = {
    "input_shape": (224, 224, 3),
    "num_classes": 5,
    "model_name": "efficientnet_b0",
    "weights": "imagenet",
    "include_top": False,
    "pooling": "avg"
}

# FAISS Configuration
FAISS_CONFIG = {
    "index_type": "IVFFlat",
    "nlist": 100,
    "nprobe": 10,
    "metric": "L2"
}

# LLaMA Configuration
LLAMA_CONFIG = {
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}

# =============================================================================
# FLOWER CLASSES
# =============================================================================

FLOWER_CLASSES = {
    0: "daisy",
    1: "dandelion", 
    2: "rose",
    3: "sunflower",
    4: "tulip"
}

FLOWER_CLASSES_FR = {
    0: "marguerite",
    1: "pissenlit",
    2: "rose", 
    3: "tournesol",
    4: "tulipe"
}

# =============================================================================
# WEB APPLICATION CONFIGURATION
# =============================================================================

# Flask Configuration
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "secret_key": os.environ.get("SECRET_KEY", "your-secret-key-here")
}

# API Configuration
API_CONFIG = {
    "max_file_size": 16 * 1024 * 1024,  # 16MB
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".bmp"},
    "upload_folder": UPLOADS_DIR
}

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Processing Configuration
PROCESSING_CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "prefetch_factor": 2
}

# Cache Configuration
CACHE_CONFIG = {
    "enable_cache": True,
    "cache_ttl": 3600,  # 1 hour
    "max_cache_size": 1000
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
}

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Load environment variables
ENV_VARS = {
    "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-key"),
    "DEBUG": os.environ.get("DEBUG", "False").lower() == "true",
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    "MODEL_PATH": os.environ.get("MODEL_PATH", str(MODELS_DIR)),
    "DATA_PATH": os.environ.get("DATA_PATH", str(DATA_DIR))
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate the configuration settings"""
    errors = []
    
    # Check if required directories exist
    required_dirs = [DATA_DIR, MODELS_DIR, SRC_DIR]
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"Required directory does not exist: {directory}")
    
    # Check if required files exist
    required_files = [SRC_DIR / "cnn_model.py", SRC_DIR / "faiss_search.py"]
    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"Required file does not exist: {file_path}")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    try:
        validate_config()
        print("✅ Configuration validation successful!")
        print(f"📁 Project root: {PROJECT_ROOT}")
        print(f"🌺 Flower classes: {list(FLOWER_CLASSES.values())}")
        print(f"🚀 Flask will run on: {FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        exit(1)
