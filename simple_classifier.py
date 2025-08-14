import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SimpleFlowerClassifier:
    """Simple flower classifier using OpenCV for basic image analysis"""
    
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
                "pink": [(140, 50, 50), (170, 255, 255)]
            },
            "sunflower": {
                "yellow": [(20, 100, 100), (35, 255, 255)],
                # Brown center of sunflower (HSV ranges for brown/orange-brown)
                "brown": [(5, 100, 40), (20, 255, 180)]
            },
            "tulip": {
                "red": [(0, 50, 50), (10, 255, 255)],
                "yellow": [(20, 100, 100), (30, 255, 255)],
                "pink": [(140, 50, 50), (170, 255, 255)]
            },
            "daisy": {
                "white": [(0, 0, 200), (180, 30, 255)],
                "yellow": [(20, 100, 100), (30, 255, 255)]
            },
            "dandelion": {
                "yellow": [(20, 100, 100), (30, 255, 255)]
            }
        }
    
    def analyze_image(self, image_path: str) -> dict:
        """Analyze flower image and return classification results"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Impossible de charger l'image")
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze colors and shapes
            color_analysis = self._analyze_colors(hsv)
            shape_analysis = self._analyze_shapes(image)
            
            # Determine flower type based on analysis
            flower_type, confidence = self._classify_flower(color_analysis, shape_analysis)
            
            # Generate description
            description = self._generate_description(flower_type, confidence, color_analysis)
            
            return {
                "class_id": flower_type,
                "flower_name_en": self.flower_classes[flower_type],
                "flower_name_fr": self.flower_classes_fr[flower_type],
                "confidence": confidence,
                "color_analysis": color_analysis,
                "shape_analysis": shape_analysis,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise
    
    def _analyze_colors(self, hsv_image) -> dict:
        """Analyze dominant colors in the image"""
        color_analysis = {}
        
        for flower_name, color_ranges in self.color_ranges.items():
            total_pixels = 0
            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for color range
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                pixel_count = cv2.countNonZero(mask)
                total_pixels += pixel_count
                
                if color_name not in color_analysis:
                    color_analysis[color_name] = 0
                color_analysis[color_name] += pixel_count
            
            color_analysis[f"{flower_name}_total"] = total_pixels
        
        return color_analysis
    
    def _analyze_shapes(self, image) -> dict:
        """Analyze basic shapes and contours in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        shape_analysis = {
            "total_contours": len(contours),
            "largest_contour_area": 0,
            "circularity": 0
        }
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            shape_analysis["largest_contour_area"] = area
            
            # Calculate circularity
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                shape_analysis["circularity"] = circularity
        
        return shape_analysis
    
    def _classify_flower(self, color_analysis: dict, shape_analysis: dict) -> tuple:
        """Classify flower based on color and shape analysis"""
        scores = {}
        
        # Score based on color analysis
        for flower_name in self.flower_classes.values():
            score = 0
            
            # Color-based scoring
            if f"{flower_name}_total" in color_analysis:
                # Normalize by number of color ranges to avoid bias toward classes
                # with more defined color bands (e.g., tulip has more than rose)
                ranges_count = max(1, len(self.color_ranges.get(flower_name, {})))
                color_score = color_analysis[f"{flower_name}_total"] / ranges_count
                score += color_score * 0.7  # 70% weight for colors
            
            # Shape-based scoring
            if flower_name == "daisy":
                # Daisies are typically highly circular
                if shape_analysis["circularity"] > 0.6:
                    score += 1000
            elif flower_name == "sunflower":
                # Sunflowers have yellow petals and a brown center; usually less perfectly circular
                # Bonus if both yellow and brown are strong
                yellow = color_analysis.get("yellow", 0)
                brown = color_analysis.get("brown", 0)
                if yellow > 1500 and brown > 800:
                    score += 1200
                # Slight penalty if too circular (to avoid confusion with dandelion)
                if shape_analysis["circularity"] > 0.65:
                    score -= 400
            elif flower_name in ["rose", "tulip"]:
                # These flowers have more complex shapes
                if shape_analysis["total_contours"] > 5:
                    score += 500
            
            scores[flower_name] = score
        
        # Find best match
        best_flower = max(scores, key=scores.get)
        best_score = scores[best_flower]
        
        # Convert flower name to class ID
        for class_id, name in self.flower_classes.items():
            if name == best_flower:
                # Calculate confidence (normalize score)
                max_possible_score = 10000  # Approximate max score
                confidence = min(best_score / max_possible_score, 0.95)
                confidence = max(confidence, 0.3)  # Minimum confidence
                
                return class_id, confidence
        
        # Default fallback
        return 2, 0.5  # Rose with 50% confidence
    
    def _generate_description(self, flower_type: int, confidence: float, color_analysis: dict) -> str:
        """Generate descriptive text about the flower"""
        flower_name_fr = self.flower_classes_fr[flower_type]
        
        # Analyze dominant colors
        dominant_colors = []
        for color_name, pixel_count in color_analysis.items():
            if color_name.endswith('_total'):
                continue
            if pixel_count > 1000:  # Threshold for significant color
                dominant_colors.append(color_name)
        
        color_desc = ""
        if dominant_colors:
            color_desc = f" avec des teintes dominantes {', '.join(dominant_colors)}"
        
        confidence_percent = confidence * 100
        
        description = f"Cette image montre une belle {flower_name_fr}{color_desc}. "
        description += f"La classification a été effectuée avec une confiance de {confidence_percent:.1f}%. "
        
        if confidence > 0.8:
            description += "La reconnaissance est très fiable grâce aux caractéristiques distinctives de cette fleur."
        elif confidence > 0.6:
            description += "La reconnaissance est assez fiable, bien que certaines caractéristiques soient moins distinctives."
        else:
            description += "La reconnaissance présente une confiance modérée, certaines caractéristiques peuvent être ambiguës."
        
        return description

# Global instance
flower_classifier = SimpleFlowerClassifier()
