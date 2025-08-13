# 🌸 CNN Model for Automatic Flower Classification
# Author: Imad SANOUSSI
# GitHub: https://github.com/ImadSANOUSSI

"""
CNN Model module for feature extraction and classification.

This module provides:
- Pre-trained CNN models (EfficientNet, ResNet, etc.)
- Feature extraction capabilities
- Transfer learning utilities
- Model training and evaluation
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
except ImportError:
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")
    tf = None
    keras = None

logger = logging.getLogger(__name__)

class FlowerCNNModel:
    """
    CNN Model for flower classification using transfer learning.
    
    Supports multiple pre-trained architectures:
    - EfficientNetB0 (default)
    - ResNet50
    - VGG16
    """
    
    def __init__(self, 
                 model_name: str = "efficientnet_b0",
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 5,
                 weights: str = "imagenet",
                 include_top: bool = False):
        """
        Initialize the CNN model.
        
        Args:
            model_name: Name of the pre-trained model
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of flower classes
            weights: Pre-trained weights to use
            include_top: Whether to include the top classification layer
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.include_top = include_top
        
        self.model = None
        self.feature_extractor = None
        self.classifier = None
        
        # Initialize the model
        self._build_model()
        
        logger.info(f"🌿 CNN Model initialized: {model_name}")
        logger.info(f"📐 Input shape: {input_shape}")
        logger.info(f"🌺 Number of classes: {num_classes}")
    
    def _build_model(self):
        """Build the CNN model architecture."""
        if tf is None:
            logger.error("❌ TensorFlow not available")
            return
        
        try:
            # Load pre-trained base model
            if self.model_name == "efficientnet_b0":
                base_model = EfficientNetB0(
                    weights=self.weights,
                    include_top=self.include_top,
                    input_shape=self.input_shape,
                    pooling="avg"
                )
            elif self.model_name == "resnet50":
                base_model = ResNet50(
                    weights=self.weights,
                    include_top=self.include_top,
                    input_shape=self.input_shape,
                    pooling="avg"
                )
            elif self.model_name == "vgg16":
                base_model = VGG16(
                    weights=self.weights,
                    include_top=self.include_top,
                    input_shape=self.input_shape,
                    pooling="avg"
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Create feature extractor
            self.feature_extractor = base_model
            
            # Add classification head
            inputs = keras.Input(shape=self.input_shape)
            x = base_model(inputs, training=False)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(512, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)
            
            # Create full model
            self.model = keras.Model(inputs, outputs)
            
            # Compile model
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            logger.info(f"✅ Model built successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error building model: {str(e)}")
            raise
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image using the CNN model.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Feature vector as numpy array
        """
        if self.feature_extractor is None:
            raise ValueError("Model not initialized")
        
        try:
            # Load and preprocess image
            img = image.load_img(image_path, target_size=self.input_shape[:2])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Extract features
            features = self.feature_extractor.predict(x, verbose=0)
            
            logger.info(f"🔍 Features extracted from: {image_path}")
            logger.info(f"📊 Feature shape: {features.shape}")
            
            return features.flatten()
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {str(e)}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict flower class for an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            # Load and preprocess image
            img = image.load_img(image_path, target_size=self.input_shape[:2])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Make prediction
            predictions = self.model.predict(x, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            result = {
                "class_id": int(predicted_class),
                "confidence": confidence,
                "predictions": predictions[0].tolist(),
                "image_path": image_path
            }
            
            logger.info(f"🎯 Prediction: Class {predicted_class}, Confidence: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {str(e)}")
            raise
    
    def fine_tune(self, 
                  train_data_dir: str,
                  validation_data_dir: str,
                  epochs: int = 10,
                  batch_size: int = 32):
        """
        Fine-tune the model on flower dataset.
        
        Args:
            train_data_dir: Directory containing training data
            validation_data_dir: Directory containing validation data
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            # Data generators
            train_datagen = image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            validation_datagen = image.ImageDataGenerator(rescale=1./255)
            
            # Load data
            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            validation_generator = validation_datagen.flow_from_directory(
                validation_data_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            # Unfreeze some layers for fine-tuning
            self.model.trainable = True
            
            # Compile with lower learning rate
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.0001),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            # Train the model
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=1
            )
            
            logger.info(f"🎯 Model fine-tuned for {epochs} epochs")
            
            return history
            
        except Exception as e:
            logger.error(f"❌ Error during fine-tuning: {str(e)}")
            raise
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            self.model.save(model_path)
            logger.info(f"💾 Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"📂 Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not initialized"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)


def create_cnn_model(model_name: str = "efficientnet_b0", **kwargs) -> FlowerCNNModel:
    """
    Factory function to create a CNN model.
    
    Args:
        model_name: Name of the pre-trained model
        **kwargs: Additional arguments for FlowerCNNModel
        
    Returns:
        Initialized FlowerCNNModel instance
    """
    return FlowerCNNModel(model_name=model_name, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = create_cnn_model(
        model_name="efficientnet_b0",
        input_shape=(224, 224, 3),
        num_classes=5
    )
    
    # Print model summary
    print("📋 Model Summary:")
    print(model.get_model_summary())
    
    print("\n✅ CNN Model module ready!")
