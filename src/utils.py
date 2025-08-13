# 🌸 Utilities for Automatic Flower Classification
# Author: Imad SANOUSSI
# GitHub: https://github.com/ImadSANOUSSI

"""
Utility functions for the Automatic Flower Classification project.

This module provides:
- Image processing utilities
- Data validation functions
- Performance monitoring
- File handling helpers
"""

import os
import logging
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import wraps
import cv2
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image with PIL
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        logger.debug(f"🖼️ Image loaded: {image_path} -> {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Error loading image {image_path}: {str(e)}")
        raise

def enhance_image(image: np.ndarray, 
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0) -> np.ndarray:
    """
    Enhance image quality using various filters.
    
    Args:
        image: Input image as numpy array
        brightness: Brightness factor (0.5 = darker, 2.0 = brighter)
        contrast: Contrast factor (0.5 = lower, 2.0 = higher)
        saturation: Saturation factor (0.0 = grayscale, 2.0 = more saturated)
        
    Returns:
        Enhanced image as numpy array
    """
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(saturation)
        
        # Convert back to numpy array
        enhanced_image = np.array(pil_image).astype(np.float32) / 255.0
        
        logger.debug(f"✨ Image enhanced: brightness={brightness}, contrast={contrast}, saturation={saturation}")
        
        return enhanced_image
        
    except Exception as e:
        logger.error(f"❌ Error enhancing image: {str(e)}")
        return image

def apply_augmentation(image: np.ndarray, 
                      rotation: float = 0,
                      flip_horizontal: bool = False,
                      flip_vertical: bool = False) -> np.ndarray:
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image as numpy array
        rotation: Rotation angle in degrees
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        
    Returns:
        Augmented image as numpy array
    """
    try:
        # Convert to PIL Image for transformations
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply rotation
        if rotation != 0:
            pil_image = pil_image.rotate(rotation, expand=True)
        
        # Apply flips
        if flip_horizontal:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        if flip_vertical:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Convert back to numpy array
        augmented_image = np.array(pil_image).astype(np.float32) / 255.0
        
        logger.debug(f"🔄 Image augmented: rotation={rotation}, flip_h={flip_horizontal}, flip_v={flip_vertical}")
        
        return augmented_image
        
    except Exception as e:
        logger.error(f"❌ Error applying augmentation: {str(e)}")
        return image

def extract_image_features(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract basic features from an image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary of extracted features
    """
    try:
        features = {}
        
        # Basic image properties
        features['shape'] = image.shape
        features['height'] = image.shape[0]
        features['width'] = image.shape[1]
        features['channels'] = image.shape[2] if len(image.shape) > 2 else 1
        
        # Color statistics
        if len(image.shape) == 3:
            features['mean_r'] = float(np.mean(image[:, :, 0]))
            features['mean_g'] = float(np.mean(image[:, :, 1]))
            features['mean_b'] = float(np.mean(image[:, :, 2]))
            features['std_r'] = float(np.std(image[:, :, 0]))
            features['std_g'] = float(np.std(image[:, :, 1]))
            features['std_b'] = float(np.std(image[:, :, 2]))
        
        # Brightness
        features['brightness'] = float(np.mean(image))
        features['contrast'] = float(np.std(image))
        
        # Edge detection
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        logger.debug(f"🔍 Features extracted: {len(features)} features")
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Error extracting features: {str(e)}")
        return {}

# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def validate_image_file(file_path: str, 
                       allowed_extensions: set = {'.jpg', '.jpeg', '.png', '.bmp'},
                       max_size_mb: int = 16) -> Tuple[bool, str]:
    """
    Validate an image file.
    
    Args:
        file_path: Path to the image file
        allowed_extensions: Set of allowed file extensions
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"File extension not allowed: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB"
        
        # Try to open image
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_flower_class(class_id: int, num_classes: int = 5) -> bool:
    """
    Validate flower class ID.
    
    Args:
        class_id: Class ID to validate
        num_classes: Total number of classes
        
    Returns:
        True if valid, False otherwise
    """
    return 0 <= class_id < num_classes

def validate_confidence(confidence: float) -> bool:
    """
    Validate confidence score.
    
    Args:
        confidence: Confidence score to validate
        
    Returns:
        True if valid, False otherwise
    """
    return 0.0 <= confidence <= 1.0

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"⏱️ {func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    return wrapper

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start a timer for a named operation."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and return execution time in milliseconds."""
        if name in self.start_times:
            execution_time = (time.time() - self.start_times[name]) * 1000
            self.metrics[name] = execution_time
            del self.start_times[name]
            return execution_time
        return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()

# =============================================================================
# FILE HANDLING UTILITIES
# =============================================================================

def ensure_directory(directory_path: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ Error creating directory {directory_path}: {str(e)}")
        return False

def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate file hash.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        File hash as hexadecimal string
    """
    try:
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"❌ Error calculating hash for {file_path}: {str(e)}")
        return ""

def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the output file
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        ensure_directory(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.debug(f"💾 JSON saved to: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving JSON to {file_path}: {str(e)}")
        return False

def load_json(file_path: str) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"📂 JSON loaded from: {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading JSON from {file_path}: {str(e)}")
        return None

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(log_level: str = "INFO", 
                 log_file: str = None,
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        log_format: Log message format
    """
    try:
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        
        logger.info(f"📝 Logging configured: level={log_level}, file={log_file}")
        
    except Exception as e:
        print(f"❌ Error setting up logging: {str(e)}")

# =============================================================================
# MAIN UTILITIES
# =============================================================================

def create_sample_data(output_dir: str = "data/sample"):
    """
    Create sample data directory structure.
    
    Args:
        output_dir: Output directory for sample data
    """
    try:
        # Create directories
        ensure_directory(output_dir)
        
        # Create flower class directories
        flower_classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
        
        for flower_class in flower_classes:
            class_dir = os.path.join(output_dir, flower_class)
            ensure_directory(class_dir)
            
            # Create placeholder file
            placeholder_file = os.path.join(class_dir, "README.md")
            with open(placeholder_file, 'w') as f:
                f.write(f"# {flower_class.title()} Sample Images\n\n")
                f.write(f"Place your {flower_class} images in this directory.\n")
                f.write("Supported formats: JPG, PNG, BMP\n")
        
        logger.info(f"📁 Sample data structure created in: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging("INFO")
    
    # Test utilities
    print("🧪 Testing utilities...")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.start_timer("test_operation")
    time.sleep(0.1)  # Simulate work
    execution_time = monitor.end_timer("test_operation")
    print(f"⏱️ Test operation took: {execution_time:.2f}ms")
    
    # Test file operations
    test_data = {"test": "data", "number": 42}
    test_file = "test_output.json"
    
    if save_json(test_data, test_file):
        loaded_data = load_json(test_file)
        print(f"📁 File operations test: {'✅' if loaded_data == test_data else '❌'}")
        
        # Clean up
        os.remove(test_file)
    
    # Create sample data
    create_sample_data()
    
    print("✅ Utilities module ready!")
