#!/usr/bin/env python3
"""
StickChef AI - Multimodal Ingredient Analyzer and Hyper-Personalized Recipe Fusion Engine
Flask backend with Streamlit frontend for food-themed AI tool.

Core Features:
- Image upload and ingredient detection using Faster R-CNN
- Recipe generation with Hugging Face text-generation
- Flavor profile visualization with Matplotlib
- Freemium monetization with Stripe integration
- Affiliate marketing for ingredients

Author: StickChef AI Team
Version: 1.1 (Monetization Ready)
"""

import sys
import argparse
import logging
import threading
import time
import json
import io
import base64
from typing import List, Dict, Optional, Tuple
import unittest
from unittest.mock import Mock, patch

# Flask and API
from flask import Flask, request, jsonify
from flask_cors import CORS

# Streamlit frontend
import streamlit as st
import requests

# Image processing and AI
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np

# Text generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Monetization
try:
    from monetization import (
        UserSubscriptionManager, 
        AffiliateManager,
        render_upgrade_prompt,
        render_usage_dashboard,
        add_affiliate_links,
        render_affiliate_section
    )
    MONETIZATION_ENABLED = True
except ImportError:
    MONETIZATION_ENABLED = False
    logging.warning("Monetization module not found. Running in free mode only.")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COCO dataset food-related classes (filtered from the full 80 classes)
COCO_FOOD_CLASSES = {
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',  # chair removed later
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush', 44: 'bottle', 45: 'wine glass',
    46: 'cup', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket'
}

# Refined food-only classes
FOOD_CLASSES = {
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 44: 'bottle',
    45: 'wine glass', 46: 'cup', 68: 'microwave', 69: 'oven', 70: 'toaster',
    72: 'refrigerator'
}

# Extended food classes for more comprehensive detection
EXTENDED_FOOD_MAPPING = {
    'apple': ['apple', 'fruit'],
    'orange': ['orange', 'citrus', 'fruit'],
    'broccoli': ['broccoli', 'vegetable', 'green vegetable'],
    'carrot': ['carrot', 'vegetable', 'root vegetable'],
    'hot dog': ['hot dog', 'sausage', 'meat'],
    'pizza': ['pizza', 'bread', 'cheese'],
    'donut': ['donut', 'pastry', 'dessert'],
    'cake': ['cake', 'dessert', 'sweet'],
    'sandwich': ['sandwich', 'bread', 'meal'],
    'bottle': ['bottle', 'condiment', 'sauce'],
    'wine glass': ['wine glass', 'beverage'],
    'cup': ['cup', 'beverage']
}

# ============================================================================
# SECTION 1: BACKEND IMAGE ANALYSIS
# ============================================================================

class IngredientDetector:
    """Handles ingredient detection from images using Faster R-CNN."""
    
    def __init__(self):
        """Initialize the detector with pre-trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained Faster R-CNN model
        try:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Faster R-CNN model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def detect_ingredients(self, image_bytes: bytes, confidence_threshold: float = 0.5) -> List[str]:
        """
        Detect ingredients from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            confidence_threshold: Minimum confidence score for detection
            
        Returns:
            List of detected ingredient names
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Process predictions
            detected_ingredients = []
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if score >= confidence_threshold and label in FOOD_CLASSES:
                    ingredient_name = FOOD_CLASSES[label]
                    if ingredient_name not in detected_ingredients:
                        detected_ingredients.append(ingredient_name)
                        logger.debug(f"Detected: {ingredient_name} (confidence: {score:.2f})")
            
            logger.info(f"Detected {len(detected_ingredients)} unique ingredients")
            return detected_ingredients
            
        except Exception as e:
            logger.error(f"Error in ingredient detection: {e}")
            raise

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Initialize detector
detector = IngredientDetector()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for server readiness."""
    return jsonify({
        'status': 'healthy',
        'service': 'StickChef AI Backend',
        'version': '1.0',
        'endpoints': ['/analyze_image', '/generate_recipe', '/generate_multiple_recipes']
    }), 200

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """
    Analyze uploaded image and return detected ingredients.
    
    Expected input: multipart/form-data with 'image' field
    Returns: JSON with detected ingredients list
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image bytes
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        # Detect ingredients
        ingredients = detector.detect_ingredients(image_bytes)
        
        # Handle no detections
        if not ingredients:
            logger.warning("No ingredients detected in image")
            return jsonify({
                'ingredients': [],
                'message': 'No ingredients detected. Please try a different image or add ingredients manually.'
            }), 200
        
        return jsonify({
            'ingredients': ingredients,
            'count': len(ingredients),
            'message': f'Successfully detected {len(ingredients)} ingredients'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /analyze_image: {e}")
        return jsonify({'error': 'Internal server error during image analysis'}), 500

# ============================================================================
# SECTION 1: UNIT TESTS FOR IMAGE ANALYSIS
# ============================================================================

class TestIngredientDetector(unittest.TestCase):
    """Unit tests for ingredient detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = IngredientDetector()
        
        # Create a simple test image (small RGB image)
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_bytes = io.BytesIO()
        self.test_image.save(self.test_image_bytes, format='JPEG')
        self.test_image_bytes = self.test_image_bytes.getvalue()
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.transform)
        self.assertIn(self.detector.device.type, ['cpu', 'cuda'])
    
    def test_detect_ingredients_with_valid_image(self):
        """Test ingredient detection with valid image bytes."""
        # This test uses a simple image, so we expect it to return a list (possibly empty)
        result = self.detector.detect_ingredients(self.test_image_bytes)
        self.assertIsInstance(result, list)
        # Note: Simple test image likely won't detect actual ingredients
    
    def test_detect_ingredients_with_invalid_image(self):
        """Test ingredient detection with invalid image bytes."""
        with self.assertRaises(Exception):
            self.detector.detect_ingredients(b'invalid_image_data')
    
    def test_empty_image_bytes(self):
        """Test detection with empty image bytes."""
        with self.assertRaises(Exception):
            self.detector.detect_ingredients(b'')

class TestFlaskRoutes(unittest.TestCase):
    """Unit tests for Flask API routes."""
    
    def setUp(self):
        """Set up test client."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_analyze_image_no_file(self):
        """Test /analyze_image with no file uploaded."""
        response = self.app.post('/analyze_image')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_analyze_image_empty_filename(self):
        """Test /analyze_image with empty filename."""
        response = self.app.post('/analyze_image', data={'image': (io.BytesIO(b''), '')})
        self.assertEqual(response.status_code, 400)
    
    @patch.object(detector, 'detect_ingredients')
    def test_analyze_image_success(self, mock_detect):
        """Test successful image analysis."""
        mock_detect.return_value = ['apple', 'orange']
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = self.app.post('/analyze_image', 
                                data={'image': (img_bytes, 'test.jpg')},
                                content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('ingredients', data)
        self.assertEqual(data['count'], 2)
        self.assertEqual(data['ingredients'], ['apple', 'orange'])

def run_section1_tests():
    """Run tests for Section 1 (Image Analysis)."""
    print("=" * 60)
    print("RUNNING SECTION 1 TESTS: Backend Image Analysis")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIngredientDetector))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFlaskRoutes))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

# ============================================================================
# SECTION 2: RECIPE GENERATION
# ============================================================================

class RecipeGenerator:
    """Handles recipe generation using GPT-2 text generation."""
    
    def __init__(self):
        """Initialize the recipe generator with GPT-2 model."""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Set pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=300,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("GPT-2 recipe generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize recipe generator: {e}")
            raise
    
    def generate_recipe(self, ingredients: List[str], cuisine_type: str = 'fusion', 
                      dietary_restrictions: List[str] = None, difficulty: str = 'medium') -> Dict:
        """Generate a single recipe with ingredients and preferences"""
        try:
            # Validate ingredients
            if not ingredients or len(ingredients) == 0:
                raise ValueError("Ingredients list cannot be empty")
            
            # Wild card fusion mode for unexpected cultural blends
            fusion_styles = [
                'Korean-Italian', 'Mexican-Japanese', 'Indian-French', 'Thai-Mediterranean',
                'Ethiopian-Scandinavian', 'Peruvian-Chinese', 'Lebanese-Mexican', 'Vietnamese-Italian'
            ]
            
            if cuisine_type == 'wild_card':
                import random
                cuisine_type = random.choice(fusion_styles)
                logging.info(f"Wild card mode selected: {cuisine_type}")
            
            # Enhanced prompt for cultural fusion
            prompt = self._build_enhanced_prompt(ingredients, cuisine_type, dietary_restrictions, difficulty)
            
            # Generate recipe text
            response = self.generator(prompt, max_length=200, num_return_sequences=1, 
                                   temperature=0.8, pad_token_id=50256, do_sample=True)
            
            recipe_text = response[0]['generated_text']
            
            # Parse recipe components
            recipe_data = self._parse_recipe_text(recipe_text, ingredients)
            
            return recipe_data
            
        except Exception as e:
            logger.error(f"Error in recipe generation: {e}")
            raise
    
    def _create_recipe_prompt(self, ingredients: str, preferences: str) -> str:
        """Create a structured prompt for recipe generation."""
        base_prompt = f"""Recipe using ingredients: {ingredients}
Preferences: {preferences}

Title: """
        
        # Add cuisine-specific context if mentioned in preferences
        if any(cuisine in preferences.lower() for cuisine in ['italian', 'chinese', 'mexican', 'indian', 'french']):
            base_prompt += f"Fusion recipe incorporating {preferences.lower()} flavors.\n"
        
        return base_prompt
    
    def _parse_recipe_text(self, generated_text: str, ingredients: List[str]) -> Dict[str, any]:
        """Parse generated recipe text into structured format."""
        # Extract recipe content from generated text
        recipe_content = generated_text.strip()
        
        # Extract title (first line)
        lines = recipe_content.split('\n')
        title = lines[0].strip() if lines else "Generated Recipe"
        
        # Clean up title
        if title.startswith("Title:"):
            title = title[6:].strip()
        
        # Generate recipe steps (simplified for demo)
        steps = self._generate_recipe_steps(recipe_content)
        
        # Generate nutrition estimate
        nutrition = self._estimate_nutrition(recipe_content)
        
        # Generate flavor profile
        flavor_profile = self._generate_flavor_profile(recipe_content, title)
        
        return {
            'title': title,
            'steps': steps,
            'nutrition': nutrition,
            'flavor_profile': flavor_profile,
            'sustainability': self._calculate_sustainability_score(ingredients),
            'generated_text': recipe_content
        }
    
    def _generate_recipe_steps(self, content: str) -> List[str]:
        """Generate recipe steps from content."""
        # Simple step generation for demo
        steps = [
            "1. Prepare and clean all ingredients",
            "2. Heat oil in a large pan or wok",
            "3. Add main ingredients and cook until tender",
            "4. Season with spices and herbs according to taste",
            "5. Serve hot and enjoy your fusion creation!"
        ]
        
        # Try to extract actual steps if present
        if "steps:" in content.lower() or "instructions:" in content.lower():
            lines = content.split('\n')
            extracted_steps = []
            for line in lines:
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    extracted_steps.append(line.strip())
            
            if extracted_steps:
                return extracted_steps[:5]  # Max 5 steps
        
        return steps
    
    def _estimate_nutrition(self, content: str) -> Dict[str, str]:
        """Estimate basic nutrition information."""
        # Simplified nutrition estimation
        nutrition = {
            'calories': '~350-450 per serving',
            'protein': 'Medium',
            'carbs': 'Medium',
            'fat': 'Low-Medium',
            'fiber': 'High',
            'sodium': 'Medium'
        }
        
        # Adjust based on content keywords
        if any(word in content.lower() for word in ['meat', 'chicken', 'beef', 'fish']):
            nutrition['protein'] = 'High'
        
        if any(word in content.lower() for word in ['pasta', 'rice', 'bread', 'potato']):
            nutrition['carbs'] = 'High'
        
        return nutrition
    
    def _generate_flavor_profile(self, content: str, title: str) -> Dict[str, float]:
        """Generate flavor profile scores based on content analysis."""
        # Initialize base scores
        flavor_scores = {
            'Sweet': 0.1,
            'Savory': 0.6,
            'Spicy': 0.2,
            'Sour': 0.1,
            'Bitter': 0.0,
            'Umami': 0.3
        }
        
        # Analyze content for flavor keywords
        content_lower = (content + " " + title).lower()
        
        # Sweet indicators
        sweet_words = ['sweet', 'sugar', 'honey', 'fruit', 'dessert', 'cake', 'vanilla']
        if any(word in content_lower for word in sweet_words):
            flavor_scores['Sweet'] = min(0.8, flavor_scores['Sweet'] + 0.4)
        
        # Spicy indicators
        spicy_words = ['spicy', 'hot', 'pepper', 'chili', 'jalapeño', 'cayenne', 'sriracha']
        if any(word in content_lower for word in spicy_words):
            flavor_scores['Spicy'] = min(0.9, flavor_scores['Spicy'] + 0.5)
        
        # Savory indicators
        savory_words = ['savory', 'salt', 'garlic', 'onion', 'herb', 'meat', 'cheese']
        if any(word in content_lower for word in savory_words):
            flavor_scores['Savory'] = min(0.9, flavor_scores['Savory'] + 0.2)
        
        # Sour indicators
        sour_words = ['sour', 'lemon', 'lime', 'vinegar', 'citrus', 'tomato']
        if any(word in content_lower for word in sour_words):
            flavor_scores['Sour'] = min(0.7, flavor_scores['Sour'] + 0.3)
        
        # Umami indicators
        umami_words = ['mushroom', 'soy', 'cheese', 'tomato', 'meat', 'fish', 'miso']
        if any(word in content_lower for word in umami_words):
            flavor_scores['Umami'] = min(0.8, flavor_scores['Umami'] + 0.3)
        
        # Normalize scores to sum to reasonable total
        total = sum(flavor_scores.values())
        if total > 1.5:  # Normalize if too high
            factor = 1.2 / total
            flavor_scores = {k: v * factor for k, v in flavor_scores.items()}
        
        return flavor_scores
    
    def _build_enhanced_prompt(self, ingredients: List[str], cuisine_type: str, 
                              dietary_restrictions: List[str] = None, difficulty: str = 'medium') -> str:
        """Build enhanced prompt for cultural fusion recipes."""
        ingredients_str = ", ".join(ingredients)
        
        # Base prompt with cultural fusion context
        prompt = f"""Create a unique {cuisine_type} fusion recipe using: {ingredients_str}

Difficulty: {difficulty}
"""
        
        # Add dietary restrictions if specified
        if dietary_restrictions:
            restrictions_str = ", ".join(dietary_restrictions)
            prompt += f"Dietary requirements: {restrictions_str}\n"
        
        # Add cuisine-specific inspiration
        if 'korean' in cuisine_type.lower():
            prompt += "Incorporate Korean flavors like gochujang, sesame, and fermented elements.\n"
        elif 'italian' in cuisine_type.lower():
            prompt += "Use Italian techniques like proper pasta cooking and herb combinations.\n"
        elif 'mexican' in cuisine_type.lower():
            prompt += "Include Mexican spices, lime, and traditional cooking methods.\n"
        elif 'japanese' in cuisine_type.lower():
            prompt += "Apply Japanese principles of umami, miso, and clean flavors.\n"
        
        prompt += "\nTitle: "
        return prompt
    
    def _calculate_sustainability_score(self, ingredients: List[str]) -> Dict[str, any]:
        """Calculate sustainability metrics for ingredients."""
        # Carbon footprint estimates (kg CO2e per kg of ingredient)
        carbon_footprint = {
            'beef': 60.0, 'lamb': 24.0, 'pork': 7.0, 'chicken': 6.0, 'fish': 5.0,
            'cheese': 11.0, 'milk': 1.9, 'egg': 4.2, 'rice': 4.0, 'pasta': 1.1,
            'bread': 0.9, 'potato': 0.5, 'tomato': 1.4, 'onion': 0.3, 'carrot': 0.4,
            'apple': 0.4, 'banana': 0.7, 'orange': 0.4, 'broccoli': 0.4, 'spinach': 0.2
        }
        
        # Seasonal availability (higher score = more sustainable)
        seasonal_score = {
            'tomato': 0.9, 'apple': 0.8, 'carrot': 0.7, 'potato': 0.9, 'onion': 0.8,
            'broccoli': 0.6, 'spinach': 0.7, 'banana': 0.4, 'orange': 0.5
        }
        
        # Calculate total carbon footprint
        total_carbon = 0
        matched_ingredients = 0
        seasonal_impact = 0
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            # Find matching carbon footprint
            for key, value in carbon_footprint.items():
                if key in ingredient_lower:
                    total_carbon += value
                    matched_ingredients += 1
                    break
            
            # Find seasonal impact
            for key, value in seasonal_score.items():
                if key in ingredient_lower:
                    seasonal_impact += value
                    break
        
        # Calculate sustainability score (0-100)
        avg_carbon = total_carbon / max(matched_ingredients, 1)
        sustainability_score = max(0, 100 - (avg_carbon * 2))  # Lower carbon = higher score
        
        # Adjust for seasonal ingredients
        if matched_ingredients > 0:
            seasonal_avg = seasonal_impact / matched_ingredients
            sustainability_score = (sustainability_score * 0.7) + (seasonal_avg * 30)
        
        return {
            'score': round(sustainability_score, 1),
            'carbon_footprint': f"{total_carbon:.1f} kg CO2e",
            'tips': self._get_sustainability_tips(avg_carbon, seasonal_impact)
        }
    
    def _get_sustainability_tips(self, carbon_level: float, seasonal_score: float) -> List[str]:
        """Get personalized sustainability tips."""
        tips = []
        
        if carbon_level > 10:
            tips.append("💡 Consider reducing meat portions or try plant-based alternatives")
        if carbon_level > 20:
            tips.append("🌱 High-impact ingredients detected - try local sourcing")
        if seasonal_score < 0.5:
            tips.append("🍂 Some ingredients may be out of season - check local availability")
        if not tips:
            tips.append("✅ Great eco-friendly ingredient choices!")
        
        tips.append("♻️ Use food scraps for composting to reduce waste")
        return tips

# Initialize recipe generator
recipe_generator = RecipeGenerator()

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    """
    Generate recipe based on ingredients and preferences.
    
    Expected input: JSON with 'ingredients' list and 'preferences' string
    Returns: JSON with recipe data including flavor profile
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'ingredients' not in data:
            return jsonify({'error': 'Missing ingredients field'}), 400
        
        ingredients = data['ingredients']
        preferences = data.get('preferences', '')
        
        # Validate ingredients
        if not isinstance(ingredients, list) or len(ingredients) == 0:
            return jsonify({'error': 'Ingredients must be a non-empty list'}), 400
        
        # Generate recipe
        recipe_data = recipe_generator.generate_recipe(ingredients, preferences)
        
        return jsonify({
            'recipe': recipe_data,
            'success': True,
            'message': 'Recipe generated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /generate_recipe: {e}")
        return jsonify({'error': 'Internal server error during recipe generation'}), 500

@app.route('/generate_multiple_recipes', methods=['POST'])
def generate_multiple_recipes():
    """
    Generate multiple recipe options based on ingredients and preferences.
    
    Expected input: JSON with 'ingredients' list and 'preferences' string
    Returns: JSON with 3 recipe options
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'ingredients' not in data:
            return jsonify({'error': 'Missing ingredients field'}), 400
        
        ingredients = data['ingredients']
        preferences = data.get('preferences', '')
        
        # Validate ingredients
        if not isinstance(ingredients, list) or len(ingredients) == 0:
            return jsonify({'error': 'Ingredients must be a non-empty list'}), 400
        
        # Generate 3 different recipes
        recipes = []
        for i in range(3):
            # Slightly modify preferences for variation
            modified_preferences = preferences
            if i == 1:
                modified_preferences += " (with a twist)"
            elif i == 2:
                modified_preferences += " (fusion style)"
            
            recipe_data = recipe_generator.generate_recipe(ingredients, modified_preferences)
            recipes.append(recipe_data)
        
        return jsonify({
            'recipes': recipes,
            'count': len(recipes),
            'success': True,
            'message': f'Generated {len(recipes)} recipe options'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /generate_multiple_recipes: {e}")
        return jsonify({'error': 'Internal server error during recipe generation'}), 500

# ============================================================================
# SECTION 2: UNIT TESTS FOR RECIPE GENERATION
# ============================================================================

class TestRecipeGenerator(unittest.TestCase):
    """Unit tests for recipe generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = RecipeGenerator()
        self.test_ingredients = ['apple', 'chicken', 'onion']
        self.test_preferences = 'Italian fusion'
    
    def test_generator_initialization(self):
        """Test that recipe generator initializes correctly."""
        self.assertIsNotNone(self.generator.tokenizer)
        self.assertIsNotNone(self.generator.model)
        self.assertIsNotNone(self.generator.generator)
    
    def test_generate_recipe_with_valid_input(self):
        """Test recipe generation with valid ingredients and preferences."""
        result = self.generator.generate_recipe(self.test_ingredients, cuisine_type=self.test_preferences)
        
        # Check required fields
        self.assertIn('title', result)
        self.assertIn('steps', result)
        self.assertIn('nutrition', result)
        self.assertIn('flavor_profile', result)
        self.assertIn('sustainability', result)
        
        # Check types
        self.assertIsInstance(result['title'], str)
        self.assertIsInstance(result['steps'], list)
        self.assertIsInstance(result['nutrition'], dict)
        self.assertIsInstance(result['flavor_profile'], dict)
        self.assertIsInstance(result['sustainability'], dict)
        
        # Check flavor profile format
        for flavor in ['Sweet', 'Savory', 'Spicy', 'Sour', 'Bitter', 'Umami']:
            self.assertIn(flavor, result['flavor_profile'])
            self.assertIsInstance(result['flavor_profile'][flavor], float)
    
    def test_generate_recipe_empty_ingredients(self):
        """Test recipe generation with empty ingredients list."""
        with self.assertRaises(Exception):
            self.generator.generate_recipe([], cuisine_type=self.test_preferences)
    
    def test_flavor_profile_parsing(self):
        """Test flavor profile generation logic."""
        # Test with spicy content
        spicy_content = "This is a spicy hot pepper recipe with chili"
        profile = self.generator._generate_flavor_profile(spicy_content, "Spicy Dish")
        self.assertGreater(profile['Spicy'], 0.4)  # Adjusted for normalization
        
        # Test with sweet content
        sweet_content = "This is a sweet dessert with honey and sugar"
        profile = self.generator._generate_flavor_profile(sweet_content, "Sweet Treat")
        self.assertGreater(profile['Sweet'], 0.3)

class TestRecipeRoutes(unittest.TestCase):
    """Unit tests for recipe generation Flask routes."""
    
    def setUp(self):
        """Set up test client."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_generate_recipe_no_json(self):
        """Test /generate_recipe with non-JSON request."""
        response = self.app.post('/generate_recipe', data='not json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_generate_recipe_missing_ingredients(self):
        """Test /generate_recipe with missing ingredients field."""
        response = self.app.post('/generate_recipe', 
                                json={'preferences': 'Italian'})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_generate_recipe_empty_ingredients(self):
        """Test /generate_recipe with empty ingredients list."""
        response = self.app.post('/generate_recipe', 
                                json={'ingredients': [], 'preferences': 'Italian'})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @patch.object(recipe_generator, 'generate_recipe')
    def test_generate_recipe_success(self, mock_generate):
        """Test successful recipe generation."""
        mock_recipe = {
            'title': 'Test Recipe',
            'steps': ['Step 1', 'Step 2'],
            'nutrition': {'calories': '300'},
            'flavor_profile': {'Sweet': 0.2, 'Savory': 0.7, 'Spicy': 0.1, 'Sour': 0.0, 'Bitter': 0.0, 'Umami': 0.3}
        }
        mock_generate.return_value = mock_recipe
        
        response = self.app.post('/generate_recipe', 
                                json={'ingredients': ['apple', 'chicken'], 'preferences': 'Italian'})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('recipe', data)
        self.assertEqual(data['recipe']['title'], 'Test Recipe')
        self.assertTrue(data['success'])
    
    @patch.object(recipe_generator, 'generate_recipe')
    def test_generate_multiple_recipes_success(self, mock_generate):
        """Test successful multiple recipe generation."""
        mock_recipe = {
            'title': 'Test Recipe',
            'steps': ['Step 1', 'Step 2'],
            'nutrition': {'calories': '300'},
            'flavor_profile': {'Sweet': 0.2, 'Savory': 0.7, 'Spicy': 0.1, 'Sour': 0.0, 'Bitter': 0.0, 'Umami': 0.3}
        }
        mock_generate.return_value = mock_recipe
        
        response = self.app.post('/generate_multiple_recipes', 
                                json={'ingredients': ['apple', 'chicken'], 'preferences': 'Italian'})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('recipes', data)
        self.assertEqual(len(data['recipes']), 3)
        self.assertEqual(data['count'], 3)
        self.assertTrue(data['success'])

def run_section2_tests():
    """Run tests for Section 2 (Recipe Generation)."""
    print("=" * 60)
    print("RUNNING SECTION 2 TESTS: Recipe Generation")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRecipeGenerator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRecipeRoutes))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

# ============================================================================
# SECTION 3: STREAMLIT FRONTEND INTEGRATION
# ============================================================================

def start_flask_server():
    """Start Flask server in background thread."""
    try:
        logger.info("Starting Flask server on localhost:5000...")
        app.run(host='localhost', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask server error: {e}")

def wait_for_flask_server(max_attempts=10):
    """Wait for Flask server to be ready."""
    import time
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:5000/', timeout=1)
            logger.info("Flask server is ready!")
            return True
        except:
            logger.debug(f"Waiting for Flask server... attempt {attempt + 1}/{max_attempts}")
            time.sleep(1)
    return False

def create_flavor_chart(flavor_profile: Dict[str, float]) -> plt.Figure:
    """Create a matplotlib bar chart for flavor profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    flavors = list(flavor_profile.keys())
    scores = list(flavor_profile.values())
    
    # Create color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Create bar chart
    bars = ax.bar(flavors, scores, color=colors[:len(flavors)])
    
    # Customize chart
    ax.set_title('Recipe Flavor Profile', fontsize=16, fontweight='bold')
    ax.set_ylabel('Intensity Score', fontsize=12)
    ax.set_xlabel('Flavor Components', fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Style the chart
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def display_recipe_card(recipe: Dict[str, any], index: int):
    """Display a recipe as a card in Streamlit."""
    with st.container():
        st.markdown(f"### 🍽️ Recipe Option {index + 1}: {recipe['title']}")
        
        # Create columns for recipe info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Instructions")
            for i, step in enumerate(recipe['steps'], 1):
                st.write(f"{i}. {step}")
        
        with col2:
            st.subheader("🥗 Nutrition")
            for key, value in recipe['nutrition'].items():
                st.write(f"**{key.capitalize()}**: {value}")
        
        # Display flavor profile chart
        if 'flavor_profile' in recipe:
            st.subheader("🌈 Flavor Profile")
            fig = create_flavor_chart(recipe['flavor_profile'])
            st.pyplot(fig)
            plt.close(fig)  # Clean up memory
        
        st.markdown("---")

def run_streamlit_app():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="StickChef AI",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B5E7A;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🍽️ StickChef AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multimodal Ingredient Analyzer & Hyper-Personalized Recipe Fusion Engine</p>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("🔧 Controls")
    
    # Initialize session state
    if 'detected_ingredients' not in st.session_state:
        st.session_state.detected_ingredients = []
    if 'recipes' not in st.session_state:
        st.session_state.recipes = []
    if 'manual_ingredients' not in st.session_state:
        st.session_state.manual_ingredients = ""
    
    # Monetization: Add usage dashboard to sidebar
    if MONETIZATION_ENABLED:
        user_email = "demo@stickchef.ai"  # In production, get from authentication
        render_usage_dashboard(user_email)
    
    # Step 1: Image Upload
    st.header("📸 Step 1: Upload Fridge/Pantry Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a photo of your fridge or pantry contents"
    )
    
    # Step 2: Image Analysis
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image for ingredients..."):
                    try:
                        # Prepare image for API
                        files = {'image': uploaded_file.getvalue()}
                        
                        # Call Flask API
                        response = requests.post('http://localhost:5000/analyze_image', files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.detected_ingredients = result.get('ingredients', [])
                            
                            st.success(f"✅ {result.get('message', 'Analysis complete')}")
                            
                            if st.session_state.detected_ingredients:
                                st.subheader("🥕 Detected Ingredients")
                                for ingredient in st.session_state.detected_ingredients:
                                    st.write(f"• {ingredient}")
                            else:
                                st.warning("No ingredients detected. You can add them manually below.")
                        else:
                            st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")
                        st.info("💡 Make sure the Flask server is running!")
    
    # Step 3: Manual Ingredient Input (Fallback)
    st.header("✏️ Step 2: Add/Edit Ingredients")
    st.write("You can add ingredients manually or edit the detected ones:")
    
    # Combine detected and manual ingredients
    current_ingredients = st.session_state.detected_ingredients.copy()
    
    manual_input = st.text_area(
        "Additional ingredients (one per line):",
        value=st.session_state.manual_ingredients,
        height=100,
        help="Add any ingredients not detected by the AI or modify the list"
    )
    
    if manual_input:
        manual_list = [ingredient.strip() for ingredient in manual_input.split('\n') if ingredient.strip()]
        current_ingredients.extend(manual_list)
        current_ingredients = list(set(current_ingredients))  # Remove duplicates
    
    if current_ingredients:
        st.subheader("🛒 Final Ingredients List")
        ingredient_cols = st.columns(3)
        for i, ingredient in enumerate(current_ingredients):
            with ingredient_cols[i % 3]:
                st.write(f"• {ingredient}")
    
    # Step 4: Preferences Input
    st.header("🎯 Step 3: Cooking Preferences")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cuisine_type = st.selectbox(
            "Cuisine Type:",
            ["Any", "Italian", "Chinese", "Mexican", "Indian", "French", "Japanese", "Thai", "Mediterranean"]
        )
        
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions:",
            ["None", "Vegetarian", "Vegan", "Gluten-free", "Dairy-free", "Low-carb", "Keto"]
        )
    
    with col2:
        cooking_time = st.slider("Max Cooking Time (minutes):", 10, 120, 30)
        difficulty = st.selectbox("Difficulty Level:", ["Easy", "Medium", "Hard"])
    
    # Additional preferences
    additional_prefs = st.text_area(
        "Additional preferences or special requests:",
        placeholder="e.g., spicy, low-sodium, kid-friendly, etc.",
        height=80
    )
    
    # Combine all preferences
    preferences = f"Cuisine: {cuisine_type}"
    if dietary_restrictions and "None" not in dietary_restrictions:
        preferences += f", Dietary: {', '.join(dietary_restrictions)}"
    preferences += f", Time: {cooking_time}min, Difficulty: {difficulty}"
    if additional_prefs:
        preferences += f", Additional: {additional_prefs}"
    
    # Step 5: Recipe Generation
    st.header("🍳 Step 4: Generate Recipes")
    
    if current_ingredients:
        # Monetization: Check usage limits
        can_generate = True
        if MONETIZATION_ENABLED:
            subscription_manager = UserSubscriptionManager()
            user_email = "demo@stickchef.ai"  # In production, get from authentication
            
            if not subscription_manager.can_generate_recipe(user_email):
                can_generate = False
                st.error("⚠️ You've reached your daily recipe limit!")
                render_upgrade_prompt(subscription_manager.get_user_plan(user_email), "recipes")
        
        if can_generate:
            # Add wild card fusion option
            use_wild_card = st.checkbox("🎲 Use Wild Card Fusion", help="Generate unexpected cultural fusion combinations!")
            
            if use_wild_card and MONETIZATION_ENABLED:
                subscription_manager = UserSubscriptionManager()
                user_email = "demo@stickchef.ai"
                
                if not subscription_manager.can_use_wild_card(user_email):
                    st.error("⚠️ You've reached your daily wild card limit!")
                    render_upgrade_prompt(subscription_manager.get_user_plan(user_email), "wild card fusions")
                    use_wild_card = False
            
            if st.button("🚀 Generate 3 Recipe Options", type="primary"):
                with st.spinner("Generating personalized recipes..."):
                    try:
                        # Update preferences for wild card
                        final_preferences = preferences
                        if use_wild_card:
                            final_preferences += ", Wild Card Fusion: True"
                        
                        # Prepare data for API
                        data = {
                            'ingredients': current_ingredients,
                            'preferences': final_preferences
                        }
                        
                        # Call Flask API
                        response = requests.post('http://localhost:5000/generate_multiple_recipes', json=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.recipes = result.get('recipes', [])
                            
                            # Monetization: Update usage after successful generation
                            if MONETIZATION_ENABLED:
                                subscription_manager = UserSubscriptionManager()
                                user_email = "demo@stickchef.ai"
                                recipe_type = "wild_card" if use_wild_card else "regular"
                                subscription_manager.increment_usage(user_email, recipe_type)
                            
                            st.success(f"✅ {result.get('message', 'Recipes generated successfully!')}")
                            
                            # Display recipes
                            if st.session_state.recipes:
                                st.header("🍽️ Your Recipe Options")
                                
                                for i, recipe in enumerate(st.session_state.recipes):
                                    # Add affiliate links to recipes
                                    if MONETIZATION_ENABLED:
                                        recipe = add_affiliate_links(recipe, current_ingredients)
                                    
                                    display_recipe_card(recipe, i)
                                    
                                    # Add affiliate section after each recipe
                                    if MONETIZATION_ENABLED:
                                        render_affiliate_section(recipe)
                                
                                # Add download option
                                st.subheader("📥 Export Recipes")
                                if st.button("Download All Recipes"):
                                    # Create downloadable content
                                    content = f"StickChef AI Recipes\n{'='*50}\n\n"
                                    content += f"Ingredients Used: {', '.join(current_ingredients)}\n"
                                    content += f"Preferences: {final_preferences}\n\n"
                                    
                                    for i, recipe in enumerate(st.session_state.recipes):
                                        content += f"Recipe {i+1}: {recipe['title']}\n"
                                        content += f"{'='*30}\n"
                                        content += "Instructions:\n"
                                        for j, step in enumerate(recipe['steps'], 1):
                                            content += f"{j}. {step}\n"
                                        content += f"\nNutrition: {recipe['nutrition']}\n"
                                        content += f"Flavor Profile: {recipe['flavor_profile']}\n"
                                        
                                        # Add sustainability info if available
                                        if 'sustainability' in recipe:
                                            content += f"Sustainability Score: {recipe['sustainability']}\n"
                                        content += "\n"
                                    
                                    st.download_button(
                                        label="📄 Download as Text File",
                                        data=content,
                                        file_name="stickchef_recipes.txt",
                                        mime="text/plain"
                                    )
                        else:
                            st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")
                        st.info("💡 Make sure the Flask server is running!")
    else:
        st.info("👆 Please add some ingredients first!")
    
    # Display existing recipes if any
    if st.session_state.recipes and not st.button("🚀 Generate 3 Recipe Options", type="primary"):
        st.header("🍽️ Your Recipe Options")
        for i, recipe in enumerate(st.session_state.recipes):
            # Add affiliate links to recipes
            if MONETIZATION_ENABLED:
                recipe = add_affiliate_links(recipe, current_ingredients)
            
            display_recipe_card(recipe, i)
            
            # Add affiliate section after each recipe
            if MONETIZATION_ENABLED:
                render_affiliate_section(recipe)
    
    # Monetization: T-shirt promotion
    if MONETIZATION_ENABLED:
        st.markdown("---")
        st.header("👕 Get the StickChef Style!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **🎨 StickChef AI T-Shirt** - Show off your fusion cooking skills!
            
            Features:
            • Custom flavor profile design
            • "Wild Card Fusion" branding  
            • Sustainability messaging
            • Premium cotton blend
            
            **Special Launch Price: $25** (Limited time!)
            """)
        
        with col2:
            if st.button("🛒 Shop Now", type="secondary"):
                st.write("🚀 Coming Soon! Pre-order available at launch.")
                st.info("Sign up for notifications when the merch store goes live!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>StickChef AI - Powered by Computer Vision & GPT-2 Text Generation</p>",
        unsafe_allow_html=True
    )

# ============================================================================
# SECTION 3: INTEGRATION TESTS
# ============================================================================

def test_flask_integration():
    """Test Flask server integration."""
    try:
        # Test image analysis endpoint
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'image': img_bytes.getvalue()}
        response = requests.post('http://localhost:5000/analyze_image', files=files)
        
        if response.status_code == 200:
            print("✅ Image analysis endpoint working")
        else:
            print(f"❌ Image analysis endpoint failed: {response.status_code}")
            return False
        
        # Test recipe generation endpoint
        data = {
            'ingredients': ['apple', 'chicken'],
            'preferences': 'Italian fusion'
        }
        
        response = requests.post('http://localhost:5000/generate_recipe', json=data)
        
        if response.status_code == 200:
            print("✅ Recipe generation endpoint working")
        else:
            print(f"❌ Recipe generation endpoint failed: {response.status_code}")
            return False
        
        # Test multiple recipes endpoint
        response = requests.post('http://localhost:5000/generate_multiple_recipes', json=data)
        
        if response.status_code == 200:
            print("✅ Multiple recipes endpoint working")
        else:
            print(f"❌ Multiple recipes endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_integration_tests():
    """Run integration tests for the full system."""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS: Full System")
    print("=" * 60)
    
    # Start Flask server in background
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Wait for server to start
    if not wait_for_flask_server():
        print("❌ Flask server failed to start")
        return False
    
    # Run integration tests
    success = test_flask_integration()
    
    if success:
        print("✅ All integration tests passed!")
        print("🚀 Ready to run: streamlit run main.py")
        print("💡 Flask server is running on http://localhost:5000")
    else:
        print("❌ Some integration tests failed")
    
    return success

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StickChef AI - Food Recognition and Recipe Generation')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--section', type=int, default=1, help='Test specific section (1=image analysis, 2=recipe generation, 3=integration)')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    args = parser.parse_args()
    
    if args.test:
        if args.section == 1:
            success = run_section1_tests()
            sys.exit(0 if success else 1)
        elif args.section == 2:
            success = run_section2_tests()
            sys.exit(0 if success else 1)
        elif args.section == 3 or args.integration:
            success = run_integration_tests()
            sys.exit(0 if success else 1)
        else:
            print(f"Section {args.section} tests not implemented")
            sys.exit(1)
    else:
        # Check if running via streamlit by looking for environment variables
        try:
            # Check if we're in a proper Streamlit runtime context
            import streamlit.runtime.scriptrunner
            ctx = streamlit.runtime.scriptrunner.get_script_run_ctx()
            if ctx is not None:
                # Running in Streamlit context - start Flask server and run app
                flask_thread = threading.Thread(target=start_flask_server, daemon=True)
                flask_thread.start()
                
                # Wait for Flask server to start
                if wait_for_flask_server():
                    logger.info("Flask server started successfully, running Streamlit app...")
                    run_streamlit_app()
                else:
                    st.error("❌ Failed to start Flask server!")
            else:
                raise Exception("Not in Streamlit context")
        except:
            # Not running in Streamlit environment - show usage info
            print("StickChef AI - Complete Implementation")
            print("=" * 50)
            print("Available sections:")
            print("  ✅ Section 1: Image Analysis (ingredient detection)")
            print("  ✅ Section 2: Recipe Generation (GPT-2 with flavor profiles)")
            print("  ✅ Section 3: Streamlit Frontend Integration")
            print()
            print("Commands:")
            print("  🧪 Test sections: python main.py --test --section [1|2|3]")
            print("  🔗 Test integration: python main.py --integration")
            print("  🚀 Run full app: streamlit run main.py")
            print()
            print("🎯 Ready for production use!") 