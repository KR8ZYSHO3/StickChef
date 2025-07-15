#!/usr/bin/env python3
"""
StickChef AI Monetization Module
Handles subscriptions, payments, and user limits for freemium model
"""

import os
import stripe
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

# Initialize Stripe (use environment variables in production)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_your_key_here")

class UserSubscriptionManager:
    """Manages user subscriptions and usage limits"""
    
    def __init__(self):
        self.plans = {
            "free": {
                "name": "Free Plan",
                "price": 0,
                "recipes_per_day": 5,
                "wild_card_limit": 1,
                "features": ["Basic recipe generation", "Flavor profiles", "Sustainability scoring"]
            },
            "pro": {
                "name": "Pro Plan", 
                "price": 4.99,
                "stripe_price_id": "price_pro_monthly",
                "recipes_per_day": 100,
                "wild_card_limit": 20,
                "features": ["Unlimited recipes", "Voice integration", "Export recipes", "Priority support"]
            },
            "premium": {
                "name": "Premium Plan",
                "price": 9.99,
                "stripe_price_id": "price_premium_monthly", 
                "recipes_per_day": 999,
                "wild_card_limit": 999,
                "features": ["Everything in Pro", "B2B menu optimization", "Custom fusion styles", "Analytics dashboard"]
            }
        }
    
    def get_user_plan(self, user_email: str) -> str:
        """Get user's current plan from session state or database"""
        if 'user_plan' not in st.session_state:
            st.session_state.user_plan = "free"
        return st.session_state.user_plan
    
    def get_user_usage(self, user_email: str) -> Dict:
        """Get user's current usage statistics"""
        if 'user_usage' not in st.session_state:
            st.session_state.user_usage = {
                "recipes_today": 0,
                "wild_cards_today": 0,
                "last_reset": datetime.now().strftime("%Y-%m-%d")
            }
        
        # Reset daily counters if new day
        usage = st.session_state.user_usage
        today = datetime.now().strftime("%Y-%m-%d")
        if usage["last_reset"] != today:
            usage["recipes_today"] = 0
            usage["wild_cards_today"] = 0
            usage["last_reset"] = today
            st.session_state.user_usage = usage
        
        return usage
    
    def can_generate_recipe(self, user_email: str) -> bool:
        """Check if user can generate a recipe"""
        plan = self.get_user_plan(user_email)
        usage = self.get_user_usage(user_email)
        limit = self.plans[plan]["recipes_per_day"]
        
        return usage["recipes_today"] < limit
    
    def can_use_wild_card(self, user_email: str) -> bool:
        """Check if user can use wild card fusion"""
        plan = self.get_user_plan(user_email)
        usage = self.get_user_usage(user_email)
        limit = self.plans[plan]["wild_card_limit"]
        
        return usage["wild_cards_today"] < limit
    
    def increment_usage(self, user_email: str, recipe_type: str = "regular"):
        """Increment user's usage counters"""
        usage = self.get_user_usage(user_email)
        usage["recipes_today"] += 1
        
        if recipe_type == "wild_card":
            usage["wild_cards_today"] += 1
        
        st.session_state.user_usage = usage
    
    def create_checkout_session(self, plan_name: str, user_email: str) -> str:
        """Create Stripe checkout session for subscription"""
        try:
            plan = self.plans[plan_name]
            
            session = stripe.checkout.Session.create(
                customer_email=user_email,
                payment_method_types=['card'],
                line_items=[{
                    'price': plan["stripe_price_id"],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url='https://your-app.com/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url='https://your-app.com/cancel',
                metadata={
                    'user_email': user_email,
                    'plan': plan_name
                }
            )
            
            return session.url
            
        except Exception as e:
            st.error(f"Error creating checkout session: {e}")
            return None

class AffiliateManager:
    """Enhanced affiliate marketing with intelligent categorization and earnings tracking"""
    
    def __init__(self, associate_tag: str = "stickchef-20"):
        self.associate_tag = associate_tag
        self.base_url = "https://www.amazon.com/s"
        self.commission_rate = 0.04  # 4% Amazon Associates
        
        # Sophisticated ingredient categorization
        self.ingredient_categories = {
            'proteins': ['chicken', 'beef', 'pork', 'salmon', 'tuna', 'fish', 'eggs', 'tofu', 'beans', 'lentils'],
            'vegetables': ['onions', 'carrots', 'celery', 'peppers', 'tomatoes', 'spinach', 'broccoli', 'mushrooms'],
            'herbs_spices': ['basil', 'oregano', 'thyme', 'cumin', 'paprika', 'garlic', 'ginger', 'cilantro'],
            'grains': ['rice', 'pasta', 'quinoa', 'oats', 'bread', 'flour', 'noodles'],
            'dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream', 'mozzarella', 'parmesan'],
            'pantry': ['olive oil', 'vinegar', 'salt', 'pepper', 'sugar', 'honey', 'soy sauce', 'coconut oil']
        }
        
        # Price estimates per category (average USD)
        self.category_prices = {
            'proteins': 15.0,
            'vegetables': 4.0,
            'herbs_spices': 6.0,
            'grains': 5.0,
            'dairy': 6.0,
            'pantry': 8.0,
            'other': 5.0
        }
    
    def _clean_ingredient(self, ingredient: str) -> str:
        """Clean ingredient name for search"""
        import re
        clean = re.sub(r'\d+\s*(cups?|tbsp|tsp|oz|lbs?|pounds?|grams?|kg)\s*', '', ingredient)
        clean = re.sub(r'^\d+\s*', '', clean)
        clean = re.sub(r'\s*\([^)]*\)', '', clean)
        return clean.strip().lower()
    
    def _get_category(self, ingredient: str) -> str:
        """Determine ingredient category"""
        ingredient_lower = self._clean_ingredient(ingredient)
        
        for category, items in self.ingredient_categories.items():
            if any(item in ingredient_lower for item in items):
                return category
        return 'other'
    
    def _estimate_price(self, ingredient: str) -> float:
        """Estimate ingredient price based on category"""
        category = self._get_category(ingredient)
        base_price = self.category_prices.get(category, 5.0)
        
        # Add some variance for realism
        import random
        variance = random.uniform(0.8, 1.2)
        return round(base_price * variance, 2)
    
    def generate_amazon_link(self, ingredient: str) -> str:
        """Generate optimized Amazon affiliate link"""
        import urllib.parse
        
        # Clean ingredient for search
        search_term = self._clean_ingredient(ingredient)
        
        # Build affiliate URL with proper parameters
        params = {
            'keywords': search_term,
            'tag': self.associate_tag,
            'ref': 'sr_st_relevanceblender',
            'qid': '1704067200',
            'sr': '1-1'
        }
        
        return f"{self.base_url}?{urllib.parse.urlencode(params)}"
    
    def generate_shopping_list(self, ingredients: list) -> Dict:
        """Generate enhanced shopping list with detailed analytics"""
        shopping_list = {
            "ingredients": [],
            "estimated_earnings": 0,
            "total_estimated_cost": 0,
            "category_breakdown": {},
            "earnings_by_category": {}
        }
        
        for ingredient in ingredients:
            category = self._get_category(ingredient)
            estimated_price = self._estimate_price(ingredient)
            affiliate_link = self.generate_amazon_link(ingredient)
            potential_earnings = estimated_price * self.commission_rate
            
            item = {
                "name": ingredient,
                "amazon_link": affiliate_link,
                "category": category.replace('_', ' ').title(),
                "estimated_price": estimated_price,
                "potential_earnings": round(potential_earnings, 2)
            }
            
            shopping_list["ingredients"].append(item)
            shopping_list["estimated_earnings"] += potential_earnings
            shopping_list["total_estimated_cost"] += estimated_price
            
            # Category breakdown
            if category not in shopping_list["category_breakdown"]:
                shopping_list["category_breakdown"][category] = 0
                shopping_list["earnings_by_category"][category] = 0
            
            shopping_list["category_breakdown"][category] += 1
            shopping_list["earnings_by_category"][category] += potential_earnings
        
        # Round final values
        shopping_list["estimated_earnings"] = round(shopping_list["estimated_earnings"], 2)
        shopping_list["total_estimated_cost"] = round(shopping_list["total_estimated_cost"], 2)
        
        return shopping_list
    
    def calculate_monthly_projection(self, daily_users: int, avg_conversion_rate: float = 0.05) -> Dict:
        """Calculate monthly affiliate earnings projection"""
        monthly_users = daily_users * 30
        purchasing_users = monthly_users * avg_conversion_rate
        avg_purchase_per_user = 25.0  # Conservative estimate
        
        monthly_sales = purchasing_users * avg_purchase_per_user
        monthly_earnings = monthly_sales * self.commission_rate
        
        return {
            "monthly_users": monthly_users,
            "purchasing_users": int(purchasing_users),
            "monthly_sales": round(monthly_sales, 2),
            "monthly_earnings": round(monthly_earnings, 2),
            "annual_projection": round(monthly_earnings * 12, 2)
        }

def render_upgrade_prompt(user_plan: str, feature: str = "recipes"):
    """Render upgrade prompt for freemium users"""
    if user_plan == "free":
        st.warning(f"🚀 **Upgrade to Pro** for unlimited {feature}!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Pro Plan - $4.99/month**")
            st.write("✅ Unlimited recipes")
            st.write("✅ 20 wild card fusions/day")
            st.write("✅ Voice integration")
            st.write("✅ Recipe export")
            
        with col2:
            st.info("**Premium Plan - $9.99/month**")
            st.write("✅ Everything in Pro")
            st.write("✅ B2B menu optimization")
            st.write("✅ Custom fusion styles")
            st.write("✅ Analytics dashboard")
        
        if st.button("🚀 Upgrade to Pro"):
            subscription_manager = UserSubscriptionManager()
            checkout_url = subscription_manager.create_checkout_session("pro", "user@example.com")
            if checkout_url:
                st.write(f"[Complete Your Upgrade]({checkout_url})")
        
        if st.button("💎 Upgrade to Premium"):
            subscription_manager = UserSubscriptionManager()
            checkout_url = subscription_manager.create_checkout_session("premium", "user@example.com")
            if checkout_url:
                st.write(f"[Complete Your Upgrade]({checkout_url})")

def render_usage_dashboard(user_email: str):
    """Render user usage dashboard"""
    subscription_manager = UserSubscriptionManager()
    plan = subscription_manager.get_user_plan(user_email)
    usage = subscription_manager.get_user_usage(user_email)
    plan_info = subscription_manager.plans[plan]
    
    st.sidebar.markdown("### 📊 Your Usage Today")
    
    # Recipe usage
    recipes_used = usage["recipes_today"]
    recipes_limit = plan_info["recipes_per_day"]
    recipe_percentage = (recipes_used / recipes_limit) * 100
    
    st.sidebar.progress(recipe_percentage / 100)
    st.sidebar.write(f"Recipes: {recipes_used}/{recipes_limit}")
    
    # Wild card usage
    wild_cards_used = usage["wild_cards_today"]
    wild_cards_limit = plan_info["wild_card_limit"]
    wild_card_percentage = (wild_cards_used / wild_cards_limit) * 100
    
    st.sidebar.progress(wild_card_percentage / 100)
    st.sidebar.write(f"Wild Cards: {wild_cards_used}/{wild_cards_limit}")
    
    # Plan info
    st.sidebar.markdown(f"**Current Plan:** {plan_info['name']}")
    if plan == "free":
        st.sidebar.button("🚀 Upgrade", key="sidebar_upgrade")

def add_affiliate_links(recipe_data: Dict, ingredients: list) -> Dict:
    """Add affiliate links to recipe data"""
    affiliate_manager = AffiliateManager()
    shopping_list = affiliate_manager.generate_shopping_list(ingredients)
    
    recipe_data["shopping_list"] = shopping_list
    recipe_data["affiliate_earnings_potential"] = shopping_list["estimated_earnings"]
    
    return recipe_data

def render_affiliate_section(recipe_data: Dict):
    """Render enhanced affiliate shopping section with analytics"""
    if "shopping_list" not in recipe_data:
        return
    
    shopping_list = recipe_data["shopping_list"]
    
    # Header with earnings summary
    st.markdown("### 🛒 Get Your Ingredients")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Total Cost", f"${shopping_list['total_estimated_cost']:.2f}")
    with col2:
        st.metric("Items", len(shopping_list["ingredients"]))
    with col3:
        st.metric("Support StickChef", f"${shopping_list['estimated_earnings']:.2f}")
    
    # Expandable shopping list
    with st.expander("🛍️ View Shopping List & Links", expanded=True):
        for item in shopping_list["ingredients"]:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{item['name']}**")
                st.caption(f"Category: {item['category']}")
            
            with col2:
                st.write(f"~${item['estimated_price']:.2f}")
            
            with col3:
                st.write(f"${item['potential_earnings']:.2f}")
            
            with col4:
                st.markdown(f"[🛒 Buy]({item['amazon_link']})")
    
    # Category breakdown
    if shopping_list.get("category_breakdown"):
        st.markdown("#### 📊 Category Breakdown")
        
        cols = st.columns(len(shopping_list["category_breakdown"]))
        for i, (category, count) in enumerate(shopping_list["category_breakdown"].items()):
            with cols[i]:
                earnings = shopping_list["earnings_by_category"][category]
                st.metric(
                    category.replace('_', ' ').title(),
                    f"{count} items",
                    f"${earnings:.2f}"
                )
    
    # Affiliate disclosure
    st.markdown("---")
    st.caption("🤝 **Affiliate Disclosure**: We earn a small commission from Amazon purchases at no extra cost to you. This helps support StickChef AI development!")

# Environment setup instructions
def setup_environment():
    """Instructions for setting up environment variables"""
    st.markdown("""
    ### 🔧 Environment Setup
    
    Add these to your `.env` file:
    
    ```bash
    STRIPE_SECRET_KEY=sk_test_your_key_here
    STRIPE_PUBLISHABLE_KEY=pk_test_your_key_here
    AMAZON_ACCESS_KEY=your_amazon_access_key
    AMAZON_SECRET_KEY=your_amazon_secret_key
    AMAZON_ASSOCIATE_TAG=stickchef-20
    ```
    """)

if __name__ == "__main__":
    # Demo usage
    st.title("StickChef AI - Monetization Demo")
    
    subscription_manager = UserSubscriptionManager()
    
    # Demo user
    user_email = "demo@stickchef.ai"
    
    # Show usage dashboard
    render_usage_dashboard(user_email)
    
    # Show upgrade prompt
    render_upgrade_prompt("free", "wild card fusions")
    
    # Show setup instructions
    setup_environment() 