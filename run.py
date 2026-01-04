#!/usr/bin/env python3
"""
Mobile Price Predictor - Run Script
Simple script to start the Flask application
"""

import os
import sys
from app import app, price_model, recommendation_rules

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 MOBILE PRICE PREDICTOR")
    print("=" * 60)
    
    # Check if models exist
    model_path = "Mobile_Price_Prediction_Recommendation/Models/mobile_price_model (1).pkl"
    rules_path = "Mobile_Price_Prediction_Recommendation/Models/fpgrowth_rules.pkl"
    
    print(f"📁 Checking model files...")
    print(f"   Price Model: {'✅ Found' if os.path.exists(model_path) else '❌ Not Found'}")
    print(f"   Rules File:  {'✅ Found' if os.path.exists(rules_path) else '❌ Not Found'}")
    
    print(f"\n🤖 Model Status:")
    print(f"   Price Model:     {'✅ Loaded' if price_model is not None else '⚠️  Using Fallback'}")
    print(f"   Recommendations: {'✅ Loaded' if recommendation_rules is not None else '⚠️  Using Fallback'}")
    
    print(f"\n🌐 Server Information:")
    print(f"   Local URL:    http://127.0.0.1:5000")
    print(f"   Network URL:  http://0.0.0.0:5000")
    print(f"   Debug Mode:   Enabled")
    
    print(f"\n📱 Features Available:")
    print(f"   ✅ Beautiful Modern UI")
    print(f"   ✅ Price Prediction")
    print(f"   ✅ Smart Recommendations") 
    print(f"   ✅ Responsive Design")
    
    print(f"\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the Flask app
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True
        )
    except KeyboardInterrupt:
        print(f"\n\n👋 Server stopped. Thank you for using Mobile Price Predictor!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)