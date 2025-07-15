#!/usr/bin/env python3
"""
StickChef AI - Stripe Setup & Testing Script
Run this after getting your Stripe API keys to test integration
"""

import os
import sys
import stripe
from colorama import init, Fore, Style

# Initialize colorama for Windows
init()

def print_status(message, status="INFO"):
    colors = {
        "INFO": Fore.BLUE,
        "SUCCESS": Fore.GREEN,
        "ERROR": Fore.RED,
        "WARNING": Fore.YELLOW
    }
    print(f"{colors.get(status, Fore.WHITE)}[{status}] {message}{Style.RESET_ALL}")

def test_stripe_connection():
    """Test Stripe API connection"""
    print_status("Testing Stripe API connection...")
    
    # Get API key from environment
    api_key = os.getenv("STRIPE_SECRET_KEY")
    
    if not api_key or api_key == "sk_test_your_key_here":
        print_status("❌ Stripe secret key not found in environment", "ERROR")
        print_status("Please set STRIPE_SECRET_KEY environment variable", "WARNING")
        return False
    
    try:
        stripe.api_key = api_key
        # Test API call
        stripe.Account.retrieve()
        print_status("✅ Stripe API connection successful!", "SUCCESS")
        return True
    except stripe.error.AuthenticationError:
        print_status("❌ Invalid Stripe API key", "ERROR")
        return False
    except Exception as e:
        print_status(f"❌ Stripe connection failed: {str(e)}", "ERROR")
        return False

def create_test_products():
    """Create test products for StickChef AI"""
    print_status("Creating test products...")
    
    try:
        # Create Pro Plan product
        pro_product = stripe.Product.create(
            name="StickChef AI Pro",
            description="100 recipes/day, 20 wild card fusions, sustainability insights"
        )
        
        pro_price = stripe.Price.create(
            unit_amount=499,  # $4.99 in cents
            currency="usd",
            recurring={"interval": "month"},
            product=pro_product.id
        )
        
        # Create Premium Plan product
        premium_product = stripe.Product.create(
            name="StickChef AI Premium",
            description="Unlimited recipes, unlimited wild cards, B2B features"
        )
        
        premium_price = stripe.Price.create(
            unit_amount=999,  # $9.99 in cents
            currency="usd",
            recurring={"interval": "month"},
            product=premium_product.id
        )
        
        print_status("✅ Test products created successfully!", "SUCCESS")
        print_status(f"Pro Plan Price ID: {pro_price.id}", "INFO")
        print_status(f"Premium Plan Price ID: {premium_price.id}", "INFO")
        
        return {
            "pro_price_id": pro_price.id,
            "premium_price_id": premium_price.id
        }
        
    except Exception as e:
        print_status(f"❌ Failed to create products: {str(e)}", "ERROR")
        return None

def test_checkout_session():
    """Test creating a checkout session"""
    print_status("Testing checkout session creation...")
    
    try:
        # Create a test checkout session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'StickChef AI Pro (Test)',
                    },
                    'unit_amount': 499,
                    'recurring': {
                        'interval': 'month',
                    },
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://your-domain.com/success',
            cancel_url='https://your-domain.com/cancel',
        )
        
        print_status("✅ Checkout session created successfully!", "SUCCESS")
        print_status(f"Session ID: {session.id}", "INFO")
        print_status(f"Checkout URL: {session.url}", "INFO")
        
        return session
        
    except Exception as e:
        print_status(f"❌ Failed to create checkout session: {str(e)}", "ERROR")
        return None

def main():
    """Main setup function"""
    print_status("🚀 StickChef AI - Stripe Setup & Testing", "INFO")
    print_status("=" * 50, "INFO")
    
    # Check if environment variables are set
    secret_key = os.getenv("STRIPE_SECRET_KEY")
    publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY")
    
    if not secret_key or secret_key == "sk_test_your_key_here":
        print_status("⚠️  Environment variables not set!", "WARNING")
        print_status("Please set the following environment variables:", "INFO")
        print_status("STRIPE_SECRET_KEY=sk_test_your_secret_key_here", "INFO")
        print_status("STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here", "INFO")
        print_status("", "INFO")
        print_status("Windows PowerShell:", "INFO")
        print_status("$env:STRIPE_SECRET_KEY='sk_test_your_key_here'", "INFO")
        print_status("$env:STRIPE_PUBLISHABLE_KEY='pk_test_your_key_here'", "INFO")
        print_status("", "INFO")
        return False
    
    print_status(f"🔑 Secret Key: {secret_key[:12]}...", "INFO")
    print_status(f"🔑 Publishable Key: {publishable_key[:12]}...", "INFO")
    print_status("", "INFO")
    
    # Test connection
    if not test_stripe_connection():
        return False
    
    # Create test products
    products = create_test_products()
    if not products:
        return False
    
    # Test checkout session
    session = test_checkout_session()
    if not session:
        return False
    
    print_status("", "INFO")
    print_status("🎉 Stripe integration is ready!", "SUCCESS")
    print_status("Next steps:", "INFO")
    print_status("1. Update main.py with your price IDs", "INFO")
    print_status("2. Deploy to Render with environment variables", "INFO")
    print_status("3. Test live payments in Stripe dashboard", "INFO")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("\n❌ Setup interrupted by user", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"❌ Unexpected error: {str(e)}", "ERROR")
        sys.exit(1) 