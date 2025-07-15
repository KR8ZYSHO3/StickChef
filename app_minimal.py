#!/usr/bin/env python3
"""
StickChef AI - Minimal Deployment Version
Lightweight version for quick deployment and Amazon Associates approval
"""

import streamlit as st
import requests
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Import monetization system
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

def create_flavor_chart(flavor_profile: Dict[str, float]) -> plt.Figure:
    """Create a simple flavor profile chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    flavors = list(flavor_profile.keys())
    values = list(flavor_profile.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FCEA2B', '#FF9F43']
    
    bars = ax.bar(flavors, values, color=colors[:len(flavors)])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Intensity')
    ax.set_title('Flavor Profile')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def generate_demo_recipe(ingredients: List[str]) -> Dict:
    """Generate a demo recipe without heavy ML models"""
    
    # Simple recipe generation for demo
    recipe_templates = [
        {
            "title": "Fusion Stir-Fry Delight",
            "steps": [
                "Heat oil in a large wok or skillet over medium-high heat",
                "Add aromatics (garlic, ginger, onions) and stir-fry for 1-2 minutes",
                "Add main ingredients and cook until tender",
                "Season with your favorite fusion spices",
                "Serve hot over rice or noodles"
            ]
        },
        {
            "title": "Mediterranean Fusion Bowl",
            "steps": [
                "Prepare a base of grains or greens",
                "Sauté vegetables with olive oil and herbs",
                "Add protein and cook until done",
                "Top with fresh herbs and a drizzle of sauce",
                "Serve with warm pita or crusty bread"
            ]
        },
        {
            "title": "Asian-Italian Fusion Pasta",
            "steps": [
                "Cook pasta according to package directions",
                "In a large pan, heat oil and add aromatics",
                "Add vegetables and stir-fry briefly",
                "Toss with cooked pasta and seasoning",
                "Garnish with fresh herbs and serve"
            ]
        }
    ]
    
    import random
    template = random.choice(recipe_templates)
    
    # Generate realistic flavor profile
    flavor_profile = {
        'Sweet': random.uniform(0.1, 0.4),
        'Savory': random.uniform(0.6, 0.9),
        'Spicy': random.uniform(0.0, 0.6),
        'Sour': random.uniform(0.0, 0.3),
        'Bitter': random.uniform(0.0, 0.2),
        'Umami': random.uniform(0.3, 0.8)
    }
    
    # Normalize to ensure realistic values
    total = sum(flavor_profile.values())
    if total > 1.5:
        factor = 1.2 / total
        flavor_profile = {k: v * factor for k, v in flavor_profile.items()}
    
    # Simple sustainability score
    sustainability = {
        'score': random.randint(70, 90),
        'carbon_footprint': f"{random.uniform(2.0, 5.0):.1f} kg CO2e",
        'tips': [
            "Choose locally sourced ingredients when possible",
            "Consider seasonal vegetables for better sustainability",
            "Reduce food waste by using all parts of ingredients"
        ]
    }
    
    return {
        'title': template['title'],
        'steps': template['steps'],
        'nutrition': {
            'calories': f"~{random.randint(300, 500)} per serving",
            'protein': 'Medium-High',
            'carbs': 'Medium',
            'fat': 'Low-Medium',
            'fiber': 'High'
        },
        'flavor_profile': flavor_profile,
        'sustainability': sustainability,
        'ingredients_used': ingredients[:5]  # Use first 5 ingredients
    }

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="StickChef AI",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    .demo-notice {
        background-color: #E8F4FD;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🍽️ StickChef AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multimodal Ingredient Analyzer & Hyper-Personalized Recipe Fusion Engine</p>', unsafe_allow_html=True)
    
    # Demo notice
    st.markdown("""
    <div class="demo-notice">
        <h3>🚀 Demo Mode</h3>
        <p>This is a lightweight demo version of StickChef AI, optimized for quick deployment. 
        Full AI-powered ingredient detection and recipe generation coming soon!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Controls")
    
    # Monetization: Add usage dashboard to sidebar
    if MONETIZATION_ENABLED:
        user_email = "demo@stickchef.ai"
        render_usage_dashboard(user_email)
    
    # Initialize session state
    if 'demo_recipes' not in st.session_state:
        st.session_state.demo_recipes = []
    
    # Step 1: Demo Image Upload
    st.header("📸 Step 1: Upload Fridge/Pantry Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a photo of your fridge or pantry contents"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("🤖 AI Analysis Coming Soon!")
            st.write("In the full version, our AI will automatically detect ingredients from your image.")
    
    # Step 2: Manual Ingredient Input
    st.header("✏️ Step 2: Enter Your Ingredients")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ingredients_input = st.text_area(
            "Enter ingredients (one per line):",
            value="chicken breast\nonions\ngarlic\nolive oil\nbasil\nrice",
            height=150,
            help="Add your available ingredients"
        )
    
    with col2:
        st.info("💡 **Example Ingredients:**")
        st.write("• Proteins: chicken, beef, tofu")
        st.write("• Vegetables: onions, carrots, peppers")
        st.write("• Herbs: basil, oregano, cilantro")
        st.write("• Pantry: olive oil, rice, pasta")
    
    # Parse ingredients
    ingredients = [ing.strip() for ing in ingredients_input.split('\n') if ing.strip()]
    
    if ingredients:
        st.subheader("🛒 Your Ingredients:")
        ingredient_cols = st.columns(min(len(ingredients), 4))
        for i, ingredient in enumerate(ingredients):
            with ingredient_cols[i % 4]:
                st.write(f"• {ingredient}")
    
    # Step 3: Generate Recipe
    st.header("🍳 Step 3: Generate Recipe")
    
    if ingredients:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            cuisine_style = st.selectbox(
                "Cuisine Style:",
                ["Fusion", "Italian", "Asian", "Mediterranean", "Mexican", "Indian"]
            )
        
        with col2:
            difficulty = st.selectbox(
                "Difficulty:",
                ["Easy", "Medium", "Hard"]
            )
        
        # Generate recipe button
        if st.button("🚀 Generate Recipe", type="primary"):
            with st.spinner("Creating your personalized recipe..."):
                # Generate demo recipe
                recipe = generate_demo_recipe(ingredients)
                
                # Add affiliate links if monetization is enabled
                if MONETIZATION_ENABLED:
                    recipe = add_affiliate_links(recipe, ingredients)
                
                st.session_state.demo_recipes = [recipe]
                st.success("✅ Recipe generated successfully!")
    
    # Display recipe
    if st.session_state.demo_recipes:
        st.header("🍽️ Your Recipe")
        
        for recipe in st.session_state.demo_recipes:
            # Recipe card
            with st.container():
                st.markdown(f"### {recipe['title']}")
                
                # Recipe content
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Instructions")
                    for i, step in enumerate(recipe['steps'], 1):
                        st.write(f"{i}. {step}")
                
                with col2:
                    st.subheader("🥗 Nutrition")
                    for key, value in recipe['nutrition'].items():
                        st.write(f"**{key.capitalize()}**: {value}")
                
                # Flavor profile
                st.subheader("🌈 Flavor Profile")
                fig = create_flavor_chart(recipe['flavor_profile'])
                st.pyplot(fig)
                plt.close(fig)
                
                # Sustainability info
                if 'sustainability' in recipe:
                    st.subheader("🌱 Sustainability")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Sustainability Score", f"{recipe['sustainability']['score']}/100")
                        st.write(f"Carbon Footprint: {recipe['sustainability']['carbon_footprint']}")
                    
                    with col2:
                        st.write("**Tips:**")
                        for tip in recipe['sustainability']['tips']:
                            st.write(f"• {tip}")
                
                # Affiliate section
                if MONETIZATION_ENABLED:
                    render_affiliate_section(recipe)
    
    # App info
    st.markdown("---")
    st.markdown("### 🔬 About StickChef AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🤖 AI-Powered**\nAdvanced ingredient detection and recipe generation")
    
    with col2:
        st.info("**🌍 Sustainable**\nCarbon footprint tracking and eco-friendly tips")
    
    with col3:
        st.info("**💡 Fusion Cooking**\nUnique cultural combinations and wild card modes")
    
    # Footer
    st.markdown("---")
    st.markdown("**StickChef AI** - Turn your fridge into a fusion laboratory! 🚀")

if __name__ == "__main__":
    main() 