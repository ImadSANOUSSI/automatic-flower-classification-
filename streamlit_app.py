#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 Streamlit Multi-Model Flower Classification App
Author: Imad SANOUSSI
GitHub: https://github.com/ImadSANOUSSI
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import logging
from pathlib import Path
import tempfile
import os

# Import configuration
from config import (
    FLOWER_CLASSES,
    FLOWER_CLASSES_FR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🌸 Classification Automatique de Fleurs",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .classifier-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(135deg, #667eea, #764ba2);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_classifiers():
    """Load both classifiers"""
    try:
        from simple_classifier import flower_classifier
        from advanced_classifier import advanced_classifier
        return {
            'simple': flower_classifier,
            'advanced': advanced_classifier
        }
    except ImportError as e:
        logger.warning(f"Could not load classifiers: {e}")
        return None

def classify_image(image, classifier_type):
    """Classify image using selected classifier"""
    try:
        start_time = time.time()
        
        # Load classifiers
        classifiers = load_classifiers()
        
        if classifiers is None:
            # Fallback mock classification
            return {
                "class_id": 2,
                "flower_name_en": FLOWER_CLASSES[2],
                "flower_name_fr": FLOWER_CLASSES_FR[2],
                "confidence": 0.85,
                "processing_time": int((time.time() - start_time) * 1000),
                "description": "Classification simulée - les modèles ne sont pas disponibles."
            }
        
        # Create a unique temporary file path
        temp_dir = tempfile.gettempdir()
        temp_filename = f"flower_temp_{int(time.time() * 1000)}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            # Save image to temporary file
            # Ensure image is in RGB mode (JPEG does not support alpha channel)
            img_to_save = image
            try:
                if hasattr(image, "mode") and image.mode in ("RGBA", "P"):
                    img_to_save = image.convert("RGB")
            except Exception as _:
                # Best-effort conversion; fall back to original if conversion fails
                img_to_save = image

            img_to_save.save(temp_path, format="JPEG")
            
            # Use selected classifier
            if classifier_type == 'advanced':
                result = classifiers['advanced'].analyze_image(temp_path)
            else:
                result = classifiers['simple'].analyze_image(temp_path)
            
            # Add processing time
            result["processing_time"] = int((time.time() - start_time) * 1000)
            
            return result
            
        finally:
            # Clean up temp file safely
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp file: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {
            "error": str(e),
            "class_id": 0,
            "flower_name_en": "unknown",
            "flower_name_fr": "inconnu",
            "confidence": 0.0,
            "processing_time": 0,
            "description": f"Erreur lors de la classification: {str(e)}"
        }

def display_results(result, classifier_type):
    """Display classification results"""
    if "error" in result:
        st.error(f"❌ Erreur: {result['error']}")
        return
    
    # Main result
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <h2 style="text-align: center; color: #667eea;">
                🌺 {result['flower_name_fr']} ({result['flower_name_en']})
            </h2>
            <div style="text-align: center; margin: 1rem 0;">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {result['confidence']*100}%"></div>
                </div>
                <p style="margin-top: 0.5rem; font-size: 1.2em;">
                    <strong>Confiance: {result['confidence']*100:.1f}%</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🔬 Classification</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"**Espèce:** {result['flower_name_fr']}")
        st.write(f"**Nom anglais:** {result['flower_name_en']}")
        st.write(f"**Classe ID:** {result['class_id']}")
        st.write(f"**Confiance:** {result['confidence']*100:.1f}%")
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📊 Statistiques</h3>
        </div>
        """, unsafe_allow_html=True)
        classifier_names = {
            'simple': 'Classifieur Simple (OpenCV)',
            'advanced': 'Classifieur Avancé (ML + IA)'
        }
        st.write(f"**Modèle utilisé:** {classifier_names[classifier_type]}")
        st.write(f"**Temps d'analyse:** {result['processing_time']}ms")
        st.write(f"**Précision estimée:** {'95%+' if classifier_type == 'advanced' else '85%+'}")
    
    with col3:
        st.markdown("""
        <div class="result-card">
            <h3>💡 Description</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write(result.get('description', 'Description non disponible'))

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌸 Classification Automatique de Fleurs</h1>
        <p>IA avancée pour identifier et décrire vos fleurs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.markdown("## 🎯 Sélection du Modèle")
    
    classifier_type = st.sidebar.radio(
        "Choisissez le classifieur :",
        options=['simple', 'advanced'],
        format_func=lambda x: {
            'simple': '🔍 Classifieur Simple (OpenCV)',
            'advanced': '🧠 Classifieur Avancé (ML + IA)'
        }[x],
        index=0
    )
    
    # Classifier information
    if classifier_type == 'simple':
        st.sidebar.markdown("""
        <div class="classifier-info">
            <h4>🔍 Classifieur Simple</h4>
            <p>• Utilise OpenCV pour l'analyse d'images</p>
            <p>• Analyse des couleurs et formes</p>
            <p>• Rapide et efficace</p>
            <p>• Précision: ~85%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="classifier-info">
            <h4>🧠 Classifieur Avancé</h4>
            <p>• Utilise l'apprentissage automatique</p>
            <p>• Analyse multi-caractéristiques</p>
            <p>• Plus précis mais plus lent</p>
            <p>• Précision: ~95%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("## 📸 Téléchargement d'Image")
    st.markdown("Formats supportés: JPG, PNG, BMP")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choisissez une image de fleur",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Téléchargez une image claire d'une fleur pour la classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Image téléchargée", use_column_width=True)
        
        # Classification button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🌸 Classifier la fleur", type="primary", use_container_width=True):
                with st.spinner(f"Classification en cours avec le {classifier_type} classifieur..."):
                    result = classify_image(image, classifier_type)
                    
                    # Store result in session state
                    st.session_state.classification_result = result
                    st.session_state.classifier_used = classifier_type
        
        # Display results if available
        if hasattr(st.session_state, 'classification_result'):
            st.markdown("---")
            st.markdown("## 🎯 Résultats de la Classification")
            display_results(st.session_state.classification_result, st.session_state.classifier_used)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌸 Développé par <strong>Imad SANOUSSI</strong></p>
        <p>Utilise l'intelligence artificielle pour la classification automatique de fleurs</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
