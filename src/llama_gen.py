# 🌸 LLaMA Generation for Automatic Flower Classification
# Author: Imad SANOUSSI
# GitHub: https://github.com/ImadSANOUSSI

"""
LLaMA Generation module for natural language descriptions.

This module provides:
- LLaMA model integration for text generation
- Flower description generation
- Natural language processing
- Context-aware text generation
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        pipeline,
        GenerationConfig
    )
    import torch
    LLAMA_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available. Install with: pip install transformers torch")
    LLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class FlowerLLAMAGenerator:
    """
    LLaMA-based text generator for flower descriptions.
    
    Provides natural language generation using:
    - LLaMA-2 models
    - Custom prompts for flowers
    - Context-aware generation
    - Multiple language support
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 use_gpu: bool = False):
        """
        Initialize LLaMA text generator.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            use_gpu: Whether to use GPU acceleration
        """
        if not LLAMA_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_gpu = use_gpu
        
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Flower-specific prompts
        self.flower_prompts = {
            "daisy": "Describe a daisy flower with its characteristic white petals and yellow center:",
            "dandelion": "Describe a dandelion flower with its bright yellow color and fluffy seed head:",
            "rose": "Describe a rose flower with its beautiful petals and distinctive fragrance:",
            "sunflower": "Describe a sunflower with its large yellow petals and dark center:",
            "tulip": "Describe a tulip flower with its cup-shaped petals and vibrant colors:"
        }
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"🤖 LLaMA Generator initialized: {model_name}")
        logger.info(f"📝 Max length: {max_length}")
        logger.info(f"🌡️ Temperature: {temperature}")
    
    def _load_model(self):
        """Load LLaMA model and tokenizer."""
        try:
            logger.info("📥 Loading LLaMA model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None,
                trust_remote_code=True
            )
            
            # Create generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1
            )
            
            logger.info("✅ LLaMA model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading LLaMA model: {str(e)}")
            # Fallback to mock generation
            self._setup_mock_generator()
    
    def _setup_mock_generator(self):
        """Setup mock generator for testing without full model."""
        logger.warning("⚠️ Using mock generator (LLaMA model not available)")
        
        self.generator = None
        
        # Mock generation function
        def mock_generate(prompt, **kwargs):
            return [{"generated_text": self._generate_mock_description(prompt)}]
        
        self.generate_text = mock_generate
    
    def _generate_mock_description(self, prompt: str) -> str:
        """Generate mock description for testing."""
        flower_type = "fleur"
        
        # Extract flower type from prompt
        for flower, flower_prompt in self.flower_prompts.items():
            if flower in prompt.lower():
                flower_type = flower
                break
        
        # Mock descriptions in French
        mock_descriptions = {
            "daisy": "Cette marguerite présente des pétales blancs délicats disposés en cercle autour d'un centre jaune vif. Sa forme simple et élégante en fait une fleur emblématique des prairies.",
            "dandelion": "Ce pissenlit se distingue par ses fleurs jaunes éclatantes qui se transforment en boules duveteuses blanches. C'est une fleur sauvage très reconnaissable.",
            "rose": "Cette rose magnifique arbore des pétales veloutés aux couleurs intenses, avec une forme élégante et un parfum envoûtant caractéristique.",
            "sunflower": "Ce tournesol impressionne par sa grande taille et ses pétales jaunes rayonnants autour d'un centre sombre et texturé.",
            "tulip": "Cette tulipe présente une forme de coupe distinctive avec des pétales colorés et une tige élégante, typique des bulbes printaniers."
        }
        
        return mock_descriptions.get(flower_type, "Cette fleur présente des caractéristiques uniques et une beauté naturelle remarquable.")
    
    def generate_flower_description(self, 
                                  flower_type: str, 
                                  confidence: float,
                                  additional_context: str = "") -> str:
        """
        Generate natural language description for a flower.
        
        Args:
            flower_type: Type of flower (daisy, rose, etc.)
            confidence: Classification confidence
            additional_context: Additional context about the image
            
        Returns:
            Generated description
        """
        try:
            # Create prompt
            prompt = self._create_flower_prompt(flower_type, confidence, additional_context)
            
            # Generate text
            if self.generator is not None:
                result = self.generator(
                    prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = result[0]["generated_text"]
                # Extract only the generated part (remove prompt)
                description = generated_text[len(prompt):].strip()
                
            else:
                # Use mock generation
                description = self._generate_mock_description(prompt)
            
            logger.info(f"📝 Generated description for {flower_type}")
            
            return description
            
        except Exception as e:
            logger.error(f"❌ Error generating description: {str(e)}")
            # Fallback to mock description
            return self._generate_mock_description(f"Describe a {flower_type} flower:")
    
    def _create_flower_prompt(self, 
                             flower_type: str, 
                             confidence: float, 
                             additional_context: str = "") -> str:
        """Create a prompt for flower description generation."""
        
        # Base prompt
        base_prompt = self.flower_prompts.get(flower_type.lower(), f"Describe a {flower_type} flower:")
        
        # Add confidence information
        confidence_text = f" (classified with {confidence*100:.1f}% confidence)"
        
        # Add context if provided
        context_text = f" Additional context: {additional_context}" if additional_context else ""
        
        # Create final prompt
        prompt = f"{base_prompt}{confidence_text}{context_text}\n\nDescription:"
        
        return prompt
    
    def generate_comparative_description(self, 
                                       flower1: str, 
                                       flower2: str,
                                       similarities: List[str],
                                       differences: List[str]) -> str:
        """
        Generate comparative description between two flowers.
        
        Args:
            flower1: First flower type
            flower2: Second flower type
            similarities: List of similarities
            differences: List of differences
            
        Returns:
            Comparative description
        """
        try:
            prompt = f"""Compare and contrast {flower1} and {flower2} flowers.

Similarities: {', '.join(similarities)}
Differences: {', '.join(differences)}

Provide a detailed comparison:"""

            if self.generator is not None:
                result = self.generator(
                    prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = result[0]["generated_text"]
                description = generated_text[len(prompt):].strip()
                
            else:
                # Mock comparative description
                description = f"Les {flower1}s et {flower2}s partagent certaines caractéristiques comme {', '.join(similarities[:2])}. Cependant, ils se distinguent par {', '.join(differences[:2])}."
            
            logger.info(f"📝 Generated comparative description: {flower1} vs {flower2}")
            
            return description
            
        except Exception as e:
            logger.error(f"❌ Error generating comparative description: {str(e)}")
            return f"Comparaison entre {flower1} et {flower2}: similitudes et différences notables."
    
    def generate_flower_care_tips(self, flower_type: str) -> str:
        """
        Generate care tips for a specific flower type.
        
        Args:
            flower_type: Type of flower
            
        Returns:
            Care tips description
        """
        try:
            prompt = f"Provide care tips and growing advice for {flower_type} flowers:"
            
            if self.generator is not None:
                result = self.generator(
                    prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = result[0]["generated_text"]
                tips = generated_text[len(prompt):].strip()
                
            else:
                # Mock care tips
                tips = f"Pour prendre soin des {flower_type}s, assurez-vous d'un arrosage régulier, d'une exposition appropriée à la lumière et d'un sol bien drainé."
            
            logger.info(f"📝 Generated care tips for {flower_type}")
            
            return tips
            
        except Exception as e:
            logger.error(f"❌ Error generating care tips: {str(e)}")
            return f"Conseils de soins pour les {flower_type}s: arrosage et entretien appropriés."
    
    def batch_generate(self, 
                       flower_types: List[str], 
                       confidences: List[float]) -> List[str]:
        """
        Generate descriptions for multiple flowers in batch.
        
        Args:
            flower_types: List of flower types
            confidences: List of confidence scores
            
        Returns:
            List of generated descriptions
        """
        try:
            descriptions = []
            
            for flower_type, confidence in zip(flower_types, confidences):
                description = self.generate_flower_description(flower_type, confidence)
                descriptions.append(description)
                
                # Small delay to avoid overwhelming the model
                time.sleep(0.1)
            
            logger.info(f"📝 Batch generation completed: {len(descriptions)} descriptions")
            
            return descriptions
            
        except Exception as e:
            logger.error(f"❌ Error in batch generation: {str(e)}")
            # Return mock descriptions as fallback
            return [self._generate_mock_description(f"Describe a {ft} flower:") 
                   for ft in flower_types]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            info = {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                "use_gpu": self.use_gpu,
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None
            }
            
            if self.model is not None:
                info["model_parameters"] = sum(p.numel() for p in self.model.parameters())
                info["model_device"] = str(next(self.model.parameters()).device)
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def save_model(self, model_path: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Save model
            self.model.save_pretrained(model_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(model_path)
            
            logger.info(f"💾 Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """Load a model from disk."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None
            )
            
            # Recreate generator
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1
            )
            
            logger.info(f"📂 Model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise


def create_llama_generator(model_name: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> FlowerLLAMAGenerator:
    """
    Factory function to create a LLaMA generator.
    
    Args:
        model_name: HuggingFace model name
        **kwargs: Additional arguments for FlowerLLAMAGenerator
        
    Returns:
        Initialized FlowerLLAMAGenerator instance
    """
    return FlowerLLAMAGenerator(model_name=model_name, **kwargs)


# Example usage
if __name__ == "__main__":
    if LLAMA_AVAILABLE:
        # Initialize LLaMA generator
        llama_gen = create_llama_generator(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            max_length=256,
            temperature=0.7
        )
        
        # Print model info
        print("🤖 Model Information:")
        print(json.dumps(llama_gen.get_model_info(), indent=2))
        
        # Generate sample description
        description = llama_gen.generate_flower_description("rose", 0.95)
        print(f"\n📝 Sample Description: {description}")
        
        print("\n✅ LLaMA Generation module ready!")
    else:
        print("❌ Transformers not available. Install with: pip install transformers torch")
