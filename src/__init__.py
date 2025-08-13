# 🌸 Automatic Flower Classification - Source Package
# Author: Imad SANOUSSI
# GitHub: https://github.com/ImadSANOUSSI

"""
Source package for Automatic Flower Classification project.

This package contains the core modules:
- cnn_model: CNN model for feature extraction
- faiss_search: FAISS for similarity search
- llama_gen: LLaMA for text generation
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Imad SANOUSSI"
__email__ = "imad.sanoussi@gmail.com"

from . import cnn_model
from . import faiss_search
from . import llama_gen
from . import utils

__all__ = [
    "cnn_model",
    "faiss_search", 
    "llama_gen",
    "utils"
]
