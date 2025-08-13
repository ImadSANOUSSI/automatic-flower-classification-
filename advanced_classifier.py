import cv2
import numpy as np
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json

# Optional TensorFlow imports (only used if a trained model exists)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    tf = None

logger = logging.getLogger(__name__)

class AdvancedFlowerClassifier:
    """Advanced flower classifier using multiple computer vision techniques"""
    
    def __init__(self):
        self.flower_classes = {
            0: "daisy",
            1: "dandelion", 
            2: "rose",
            3: "sunflower",
            4: "tulip"
        }
        
        self.flower_classes_fr = {
            0: "marguerite",
            1: "pissenlit",
            2: "rose", 
            3: "tournesol",
            4: "tulipe"
        }
        
        # Color ranges for different flower types (HSV)
        self.color_ranges = {
            "rose": {
                "red": [(0, 50, 50), (10, 255, 255)],
                "pink": [(140, 50, 50), (170, 255, 255)],
                "dark_red": [(0, 100, 100), (10, 255, 255)]
            },
            "sunflower": {
                "yellow": [(20, 100, 100), (30, 255, 255)],
                "orange": [(10, 100, 100), (20, 255, 255)]
            },
            "tulip": {
                "red": [(0, 50, 50), (10, 255, 255)],
                "yellow": [(20, 100, 100), (30, 255, 255)],
                "pink": [(140, 50, 50), (170, 255, 255)],
                "purple": [(130, 50, 50), (150, 255, 255)]
            },
            "daisy": {
                "white": [(0, 0, 200), (180, 30, 255)],
                "yellow": [(20, 100, 100), (30, 255, 255)],
                "cream": [(15, 30, 200), (25, 100, 255)]
            },
            "dandelion": {
                "yellow": [(20, 100, 100), (30, 255, 255)],
                "light_yellow": [(20, 50, 150), (30, 150, 255)]
            }
        }
        
        # Initialize classical ML fallback model
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "flower_classifier_model.pkl"
        
        # Paths for optional deep model
        self.models_dir = Path("models")
        self.tf_model_path = self.models_dir / "flower_efficientnet.keras"
        self.labels_path = self.models_dir / "flower_labels.json"
        self.tf_model = None
        self.class_names = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded existing flower classification model")
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                logger.info("Created new RandomForest classifier")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Try loading TensorFlow model if available
        try:
            if tf is not None and self.tf_model_path.exists() and self.labels_path.exists():
                self.tf_model = tf.keras.models.load_model(self.tf_model_path)
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    self.class_names = json.load(f)
                # Ensure mapping is list index -> name
                if isinstance(self.class_names, dict):
                    # Convert dict with string keys to list based on sorted keys
                    sorted_items = sorted(self.class_names.items(), key=lambda x: int(x[0]))
                    self.class_names = [name for _, name in sorted_items]
                logger.info(f"Loaded TF model with {len(self.class_names)} classes from {self.tf_model_path}")
            else:
                if tf is None:
                    logger.info("TensorFlow not available; using classical features pipeline")
                else:
                    logger.info("No trained TF model found; using classical features pipeline")
        except Exception as e:
            logger.warning(f"Could not load TF model: {e}")
    
    def analyze_image(self, image_path: str) -> dict:
        """Analyze flower image and return classification results"""
        try:
            # If a trained TF model is available, prefer it
            if self.tf_model is not None and self.class_names is not None:
                # Load with PIL to ensure RGB
                from PIL import Image
                pil_img = Image.open(image_path).convert("RGB").resize((224, 224))
                x = np.array(pil_img, dtype=np.float32)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = self.tf_model.predict(x, verbose=0)[0]
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))
                flower_name_en = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                flower_name_fr = self._to_french(flower_name_en)
                description = self._generate_description(class_id, confidence, {"model":"tf_efficientnet"})
                return {
                    "class_id": class_id,
                    "flower_name_en": flower_name_en,
                    "flower_name_fr": flower_name_fr,
                    "confidence": confidence,
                    "description": description,
                    "features_used": ["deep_learning"]
                }
            
            # Otherwise fallback to classical pipeline
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Impossible de charger l'image")
            
            # Resize image for consistent analysis
            image = cv2.resize(image, (224, 224))
            
            # Extract multiple features
            color_features = self._extract_color_features(image)
            shape_features = self._extract_shape_features(image)
            texture_features = self._extract_texture_features(image)
            
            # Combine all features
            all_features = np.concatenate([color_features, shape_features, texture_features])
            
            # Make prediction
            prediction, confidence = self._classify_flower(all_features)
            
            # Generate detailed analysis
            analysis_details = self._generate_analysis_details(image, color_features, shape_features, texture_features)
            
            # Generate description
            description = self._generate_description(prediction, confidence, analysis_details)
            
            return {
                "class_id": prediction,
                "flower_name_en": self.flower_classes[prediction],
                "flower_name_fr": self.flower_classes_fr[prediction],
                "confidence": confidence,
                "analysis_details": analysis_details,
                "description": description,
                "features_used": ["couleurs", "formes", "textures", "machine_learning"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise

    def _to_french(self, name_en: str) -> str:
        mapping = {
            "daisy": "marguerite",
            "dandelion": "pissenlit",
            "rose": "rose",
            "sunflower": "tournesol",
            "tulip": "tulipe"
        }
        return mapping.get(name_en.lower(), name_en)
    
    def _extract_color_features(self, image) -> np.ndarray:
        """Extract comprehensive color features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # HSV color analysis
        for flower_name, color_ranges in self.color_ranges.items():
            total_pixels = 0
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = cv2.countNonZero(mask)
                total_pixels += pixel_count
                features.append(pixel_count / (224 * 224))  # Normalize
            
            features.append(total_pixels / (224 * 224))
        
        # Color histograms
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist[::8])  # Sample every 8th bin
        
        # Dominant colors
        pixels = image.reshape(-1, 3)
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            features.extend(dominant_colors.flatten() / 255)  # Normalize
        except:
            features.extend([0] * 15)  # Fallback
        
        return np.array(features)
    
    def _extract_shape_features(self, image) -> np.ndarray:
        """Extract shape and contour features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with multiple thresholds
        edges_low = cv2.Canny(blurred, 30, 100)
        edges_high = cv2.Canny(blurred, 100, 200)
        
        # Find contours
        contours_low, _ = cv2.findContours(edges_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_high, _ = cv2.findContours(edges_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        # Contour analysis
        for contours in [contours_low, contours_high]:
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                features.extend([
                    len(contours) / 100,  # Number of contours
                    area / (224 * 224),   # Largest contour area
                    perimeter / (224 * 4), # Perimeter
                    4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0,  # Circularity
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Shape moments
        if contours_low:
            moments = cv2.moments(contours_low[0])
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                features.extend([cx / 224, cy / 224])  # Centroid
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def _extract_texture_features(self, image) -> np.ndarray:
        """Extract texture features using GLCM-like approach"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Local Binary Pattern approximation
        lbp = self._local_binary_pattern(gray)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        features.extend(hist[::4])  # Sample every 4th bin
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        features.extend([
            np.mean(gradient_magnitude) / 255,
            np.std(gradient_magnitude) / 255,
            np.mean(gradient_direction),
            np.std(gradient_direction)
        ])
        
        # Haralick-like features (simplified)
        # Variance of pixel differences
        diff_x = np.diff(gray, axis=1)
        diff_y = np.diff(gray, axis=0)
        
        features.extend([
            np.var(diff_x) / 255**2,
            np.var(diff_y) / 255**2,
            np.mean(np.abs(diff_x)) / 255,
            np.mean(np.abs(diff_y)) / 255
        ])
        
        return np.array(features)
    
    def _local_binary_pattern(self, image):
        """Compute Local Binary Pattern"""
        patterns = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] > center) << 7
                code |= (image[i-1, j] > center) << 6
                code |= (image[i-1, j+1] > center) << 5
                code |= (image[i, j+1] > center) << 4
                code |= (image[i+1, j+1] > center) << 3
                code |= (image[i+1, j] > center) << 2
                code |= (image[i+1, j-1] > center) << 1
                code |= (image[i, j-1] > center) << 0
                patterns[i, j] = code
        return patterns
    
    def _classify_flower(self, features: np.ndarray) -> tuple:
        """Classify flower using ML model and fallback methods"""
        try:
            # Try ML model first
            if self.model is not None:
                # Reshape features for sklearn
                features_2d = features.reshape(1, -1)
                
                # Make prediction
                prediction = self.model.predict(features_2d)[0]
                
                # Get prediction probabilities
                proba = self.model.predict_proba(features_2d)[0]
                confidence = proba[prediction]
                
                return int(prediction), confidence
                
        except Exception as e:
            logger.warning(f"ML model failed: {e}")
        
        # Fallback to rule-based classification
        return self._rule_based_classification(features)
    
    def _rule_based_classification(self, features: np.ndarray) -> tuple:
        """Rule-based classification as fallback"""
        # Simple scoring based on color features
        scores = np.zeros(5)
        
        # Color-based scoring (first 25 features are color-related)
        color_features = features[:25]
        
        # Score each flower type based on color presence
        for i, flower_name in enumerate(self.flower_classes.values()):
            if flower_name == "rose":
                # Red/pink colors
                scores[i] += color_features[0] + color_features[1] + color_features[2]
            elif flower_name == "sunflower":
                # Yellow/orange colors
                scores[i] += color_features[3] + color_features[4]
            elif flower_name == "tulip":
                # Multiple colors
                scores[i] += color_features[5] + color_features[6] + color_features[7] + color_features[8]
            elif flower_name == "daisy":
                # White/yellow colors
                scores[i] += color_features[9] + color_features[10] + color_features[11]
            elif flower_name == "dandelion":
                # Yellow colors
                scores[i] += color_features[12] + color_features[13]
        
        # Find best match
        best_class = np.argmax(scores)
        confidence = min(scores[best_class] / np.max(scores), 0.9)
        confidence = max(confidence, 0.3)  # Minimum confidence
        
        return best_class, confidence
    
    def _generate_analysis_details(self, image, color_features, shape_features, texture_features) -> dict:
        """Generate detailed analysis information"""
        return {
            "color_analysis": {
                "dominant_colors": self._get_dominant_colors(color_features),
                "color_variety": np.std(color_features[:25]),
                "brightness": np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255
            },
            "shape_analysis": {
                "complexity": shape_features[0] if len(shape_features) > 0 else 0,
                "symmetry": shape_features[3] if len(shape_features) > 3 else 0,
                "area_coverage": shape_features[1] if len(shape_features) > 1 else 0
            },
            "texture_analysis": {
                "smoothness": 1 - (texture_features[0] if len(texture_features) > 0 else 0),
                "regularity": 1 - (texture_features[4] if len(texture_features) > 4 else 0)
            }
        }
    
    def _get_dominant_colors(self, color_features) -> list:
        """Extract dominant color information"""
        # This is a simplified version - in practice you'd analyze the actual color distributions
        colors = ["rouge", "rose", "jaune", "orange", "blanc", "violet"]
        dominant = []
        
        for i, color in enumerate(colors):
            if i < len(color_features) and color_features[i] > 0.1:
                dominant.append(color)
        
        return dominant if dominant else ["mixte"]
    
    def _generate_description(self, flower_type: int, confidence: float, analysis_details: dict) -> str:
        """Generate comprehensive description of the flower"""
        flower_name_fr = self.flower_classes_fr[flower_type]
        confidence_percent = confidence * 100
        
        # Base description
        description = f"Cette image montre une belle {flower_name_fr}. "
        
        # Add color information
        if analysis_details["color_analysis"]["dominant_colors"]:
            colors = ", ".join(analysis_details["color_analysis"]["dominant_colors"])
            description += f"Les couleurs dominantes sont {colors}. "
        
        # Add shape information
        if analysis_details["shape_analysis"]["complexity"] > 0.5:
            description += "La forme de la fleur présente une complexité intéressante. "
        elif analysis_details["shape_analysis"]["symmetry"] > 0.6:
            description += "La fleur présente une symétrie remarquable. "
        
        # Add texture information
        if analysis_details["texture_analysis"]["smoothness"] > 0.7:
            description += "Les pétales ont une texture lisse et délicate. "
        
        # Add confidence information
        description += f"La classification a été effectuée avec une confiance de {confidence_percent:.1f}%. "
        
        if confidence > 0.8:
            description += "La reconnaissance est très fiable grâce aux caractéristiques distinctives de cette fleur."
        elif confidence > 0.6:
            description += "La reconnaissance est assez fiable, bien que certaines caractéristiques soient moins distinctives."
        else:
            description += "La reconnaissance présente une confiance modérée, certaines caractéristiques peuvent être ambiguës."
        
        return description

# Global instance
advanced_classifier = AdvancedFlowerClassifier()
