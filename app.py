from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import threading
from functools import lru_cache
import time

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

price_model = None
recommendation_rules = None
mobile_database = None
model_lock = threading.Lock()
model_loaded = False

def create_smart_model():
    class SmartMobilePriceModel:
        def __init__(self):
            self.brand_base_prices = {
                'Apple': 45000, 'Samsung': 25000, 'OnePlus': 30000, 'Google': 40000,
                'Xiaomi': 15000, 'Realme': 12000, 'Vivo': 18000, 'Oppo': 20000,
                'Huawei': 25000, 'Nokia': 10000, 'Poco': 12000, 'Motorola': 15000,
                'Honor': 16000, 'iQOO': 22000, 'Asus': 28000, 'LG': 20000,
                'Lenovo': 14000
            }
            self.os_multipliers = {'iOS': 1.4, 'Android': 1.0}
            self.ram_price_per_gb = 1500
            self.storage_price_per_gb = 80
            self.screen_size_factor = 2000
            self.battery_factor = 3
            self.year_factors = {
                2024: 1.2, 2023: 1.1, 2022: 1.0, 2021: 0.85, 
                2020: 0.7, 2019: 0.6, 2018: 0.5, 2017: 0.4, 
                2016: 0.35, 2015: 0.3, 2014: 0.25
            }
            self.processor_multipliers = {
                'octa-core': 1.2, 'quad-core': 0.8, 'hexa-core': 1.0, 
                'Unknown_Processor': 1.0
            }
        
        def predict(self, X):
            predictions = []
            for _, row in X.iterrows():
                brand = row.get('Brand', 'Xiaomi')
                base_price = self.brand_base_prices.get(brand, 15000)
                os = row.get('operating_system', 'Android')
                os_mult = self.os_multipliers.get(os, 1.0)
                price = base_price * os_mult
                ram = row.get('RAM', 4096)
                if ram > 100:
                    ram_gb = ram / 1024
                else:
                    ram_gb = ram
                price += ram_gb * self.ram_price_per_gb
                storage = row.get('Internal_storage(GB)', 64)
                price += storage * self.storage_price_per_gb
                screen_size = row.get('Screen-size', 6.0)
                price += (screen_size - 5.0) * self.screen_size_factor
                battery = row.get('Battery(mah)', 4000)
                price += (battery - 3000) * self.battery_factor
                year = int(row.get('Release_year', 2022))
                year_mult = self.year_factors.get(year, 0.5)
                price *= year_mult
                processor = row.get('Processor', 'octa-core')
                proc_mult = self.processor_multipliers.get(processor, 1.0)
                price *= proc_mult
                final_price = max(3000, min(150000, price))
                predictions.append(final_price)
            return np.array(predictions)
    return SmartMobilePriceModel()

def load_models():
    global price_model, recommendation_rules, mobile_database, model_loaded
    
    with model_lock:
        if model_loaded:
            return price_model, recommendation_rules
            
        try:
            print("Loading models...")
            
            # First try to load the pre-trained model
            model_path = "Models/mobile_price_model (1).pkl"
            if os.path.exists(model_path):
                print("Loading pre-trained model from pickle file...")
                with open(model_path, "rb") as f:
                    price_model = pickle.load(f)
                print("Pre-trained model loaded successfully!")
            else:
                # If pickle file doesn't exist, train a new model
                data_path = "data/cleaned_data1 (1).csv"
                if os.path.exists(data_path):
                    print("Training new model from data...")
                    df = pd.read_csv(data_path)
                    mobile_database = df
                    X = df[['Brand','operating_system','Processor','Release_year',
                            'Screen-size','Internal_storage(GB)','Battery(mah)','RAM']]
                    y = df['Price_in_India']
                    categorical_cols = ['Brand','operating_system','Processor']
                    preprocessor = ColumnTransformer([
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ], remainder='passthrough')
                    pipeline = Pipeline([
                        ('preprocess', preprocessor),
                        ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
                    ])
                    print("Training model...")
                    pipeline.fit(X, y)
                    price_model = pipeline
                    print("Model trained successfully!")
                else:
                    print("No data file found, using fallback model")
                    price_model = create_smart_model()
            
            # Load recommendation rules
            rules_path = "Models/fpgrowth_rules.pkl"
            if os.path.exists(rules_path):
                try:
                    with open(rules_path, "rb") as f:
                        recommendation_rules = pickle.load(f)
                    print("Recommendation rules loaded!")
                except Exception:
                    recommendation_rules = None
            else:
                recommendation_rules = None
                
            model_loaded = True
            return price_model, recommendation_rules
            
        except Exception as e:
            print(f"Error loading models: {e}")
            price_model = create_smart_model()
            model_loaded = True
            return price_model, None

def create_mobile_recommendations(budget_min=0, budget_max=100000, brand='', os='', min_ram=0, min_storage=0, min_battery=0):
    # Complete mobile database with all brands
    all_recommendations = {
        'Apple': [
            {'recommended_item': 'iPhone 15 Pro Max', 'specifications': '6.7" Display, 8GB RAM, 256GB Storage, 4441mAh Battery', 'price': '₹1,59,900', 'price_range': 'Ultra Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 15 Pro', 'specifications': '6.1" Display, 8GB RAM, 128GB Storage, 3274mAh Battery', 'price': '₹1,34,900', 'price_range': 'Ultra Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 15', 'specifications': '6.1" Display, 6GB RAM, 128GB Storage, 3349mAh Battery', 'price': '₹79,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 14 Pro Max', 'specifications': '6.7" Display, 6GB RAM, 128GB Storage, 4323mAh Battery', 'price': '₹1,39,900', 'price_range': 'Ultra Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 14 Pro', 'specifications': '6.1" Display, 6GB RAM, 128GB Storage, 3200mAh Battery', 'price': '₹1,29,900', 'price_range': 'Ultra Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 14 Plus', 'specifications': '6.7" Display, 6GB RAM, 128GB Storage, 4325mAh Battery', 'price': '₹89,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 14', 'specifications': '6.1" Display, 6GB RAM, 128GB Storage, 3279mAh Battery', 'price': '₹79,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 13 Pro Max', 'specifications': '6.7" Display, 6GB RAM, 128GB Storage, 4352mAh Battery', 'price': '₹1,19,900', 'price_range': 'Ultra Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 13 Pro', 'specifications': '6.1" Display, 6GB RAM, 128GB Storage, 3095mAh Battery', 'price': '₹1,09,900', 'price_range': 'Ultra Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 13', 'specifications': '6.1" Display, 4GB RAM, 128GB Storage, 3240mAh Battery', 'price': '₹69,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 13 mini', 'specifications': '5.4" Display, 4GB RAM, 128GB Storage, 2438mAh Battery', 'price': '₹59,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 12 Pro Max', 'specifications': '6.7" Display, 6GB RAM, 128GB Storage, 3687mAh Battery', 'price': '₹99,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 12 Pro', 'specifications': '6.1" Display, 6GB RAM, 128GB Storage, 2815mAh Battery', 'price': '₹89,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 12', 'specifications': '6.1" Display, 4GB RAM, 64GB Storage, 2815mAh Battery', 'price': '₹59,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 12 mini', 'specifications': '5.4" Display, 4GB RAM, 64GB Storage, 2227mAh Battery', 'price': '₹49,900', 'price_range': 'Mid-range', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone SE 3rd Gen', 'specifications': '4.7" Display, 4GB RAM, 64GB Storage, 2018mAh Battery', 'price': '₹43,900', 'price_range': 'Mid-range', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 11 Pro Max', 'specifications': '6.5" Display, 4GB RAM, 64GB Storage, 3969mAh Battery', 'price': '₹79,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 11 Pro', 'specifications': '5.8" Display, 4GB RAM, 64GB Storage, 3046mAh Battery', 'price': '₹69,900', 'price_range': 'Premium', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 11', 'specifications': '6.1" Display, 4GB RAM, 64GB Storage, 3110mAh Battery', 'price': '₹49,900', 'price_range': 'Mid-range', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone XR', 'specifications': '6.1" Display, 3GB RAM, 64GB Storage, 2942mAh Battery', 'price': '₹39,900', 'price_range': 'Mid-range', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone XS', 'specifications': '5.8" Display, 4GB RAM, 64GB Storage, 2658mAh Battery', 'price': '₹35,900', 'price_range': 'Mid-range', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone X (Refurbished)', 'specifications': '5.8" Display, 3GB RAM, 64GB Storage, 2716mAh Battery', 'price': '₹29,900', 'price_range': 'Budget', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 8 Plus (Refurbished)', 'specifications': '5.5" Display, 3GB RAM, 64GB Storage, 2691mAh Battery', 'price': '₹25,900', 'price_range': 'Budget', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 8 (Refurbished)', 'specifications': '4.7" Display, 2GB RAM, 64GB Storage, 1821mAh Battery', 'price': '₹19,900', 'price_range': 'Budget', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 7 Plus (Refurbished)', 'specifications': '5.5" Display, 3GB RAM, 32GB Storage, 2900mAh Battery', 'price': '₹15,900', 'price_range': 'Budget', 'brand': 'Apple', 'os': 'iOS'},
            {'recommended_item': 'iPhone 7 (Refurbished)', 'specifications': '4.7" Display, 2GB RAM, 32GB Storage, 1960mAh Battery', 'price': '₹12,900', 'price_range': 'Budget', 'brand': 'Apple', 'os': 'iOS'}
        ],
        'Samsung': [
            {'recommended_item': 'Samsung Galaxy S24 Ultra', 'specifications': '6.8" Display, 12GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹1,29,999', 'price_range': 'Ultra Premium', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy S23 Ultra', 'specifications': '6.8" Display, 12GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹1,24,999', 'price_range': 'Ultra Premium', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy Z Fold 5', 'specifications': '7.6" Foldable Display, 12GB RAM, 256GB Storage, 4400mAh Battery', 'price': '₹1,54,999', 'price_range': 'Ultra Premium', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy S23 FE', 'specifications': '6.4" Display, 8GB RAM, 128GB Storage, 4500mAh Battery', 'price': '₹59,999', 'price_range': 'Premium', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy A54 5G', 'specifications': '6.4" Display, 8GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹38,999', 'price_range': 'Mid-range', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy A34 5G', 'specifications': '6.6" Display, 8GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹30,999', 'price_range': 'Mid-range', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy M54 5G', 'specifications': '6.7" Display, 8GB RAM, 256GB Storage, 6000mAh Battery', 'price': '₹26,999', 'price_range': 'Mid-range', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy A24', 'specifications': '6.5" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹22,999', 'price_range': 'Budget', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy M34 5G', 'specifications': '6.5" Display, 6GB RAM, 128GB Storage, 6000mAh Battery', 'price': '₹18,999', 'price_range': 'Budget', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy A14 5G', 'specifications': '6.6" Display, 4GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹16,499', 'price_range': 'Budget', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy M14 5G', 'specifications': '6.6" Display, 4GB RAM, 128GB Storage, 6000mAh Battery', 'price': '₹13,490', 'price_range': 'Budget', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy A04s', 'specifications': '6.5" Display, 4GB RAM, 64GB Storage, 5000mAh Battery', 'price': '₹11,499', 'price_range': 'Budget', 'brand': 'Samsung', 'os': 'Android'},
            {'recommended_item': 'Samsung Galaxy A04e', 'specifications': '6.5" Display, 3GB RAM, 32GB Storage, 5000mAh Battery', 'price': '₹9,499', 'price_range': 'Budget', 'brand': 'Samsung', 'os': 'Android'}
        ],
        'OnePlus': [
            {'recommended_item': 'OnePlus 12', 'specifications': '6.82" Display, 16GB RAM, 512GB Storage, 5400mAh Battery', 'price': '₹69,999', 'price_range': 'Premium', 'brand': 'OnePlus', 'os': 'Android'},
            {'recommended_item': 'OnePlus 11', 'specifications': '6.7" Display, 12GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹56,999', 'price_range': 'Premium', 'brand': 'OnePlus', 'os': 'Android'},
            {'recommended_item': 'OnePlus 11R', 'specifications': '6.74" Display, 8GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹39,999', 'price_range': 'Mid-range', 'brand': 'OnePlus', 'os': 'Android'},
            {'recommended_item': 'OnePlus Nord CE 3 Lite', 'specifications': '6.72" Display, 8GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹19,999', 'price_range': 'Budget', 'brand': 'OnePlus', 'os': 'Android'},
            {'recommended_item': 'OnePlus Nord CE 3', 'specifications': '6.7" Display, 8GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹26,999', 'price_range': 'Mid-range', 'brand': 'OnePlus', 'os': 'Android'}
        ],
        'Oppo': [
            {'recommended_item': 'Oppo Find X6 Pro', 'specifications': '6.82" Display, 16GB RAM, 512GB Storage, 5000mAh Battery', 'price': '₹89,999', 'price_range': 'Premium', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo Reno 10 Pro+', 'specifications': '6.74" Display, 12GB RAM, 256GB Storage, 4700mAh Battery', 'price': '₹54,999', 'price_range': 'Premium', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo Reno 10 Pro', 'specifications': '6.7" Display, 12GB RAM, 256GB Storage, 4600mAh Battery', 'price': '₹39,999', 'price_range': 'Mid-range', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo Reno 8T 5G', 'specifications': '6.43" Display, 8GB RAM, 128GB Storage, 4800mAh Battery', 'price': '₹29,999', 'price_range': 'Mid-range', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo F23 5G', 'specifications': '6.72" Display, 8GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹24,999', 'price_range': 'Mid-range', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo A98 5G', 'specifications': '6.72" Display, 8GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹23,999', 'price_range': 'Budget', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo A78 5G', 'specifications': '6.56" Display, 8GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹18,999', 'price_range': 'Budget', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo A58 5G', 'specifications': '6.72" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹15,999', 'price_range': 'Budget', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo A17k', 'specifications': '6.56" Display, 3GB RAM, 64GB Storage, 5000mAh Battery', 'price': '₹12,999', 'price_range': 'Budget', 'brand': 'Oppo', 'os': 'Android'},
            {'recommended_item': 'Oppo A16k', 'specifications': '6.52" Display, 3GB RAM, 32GB Storage, 4230mAh Battery', 'price': '₹10,999', 'price_range': 'Budget', 'brand': 'Oppo', 'os': 'Android'}
        ],
        'Xiaomi': [
            {'recommended_item': 'Xiaomi 13 Pro', 'specifications': '6.73" Display, 12GB RAM, 256GB Storage, 4820mAh Battery', 'price': '₹79,999', 'price_range': 'Premium', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi 13', 'specifications': '6.36" Display, 8GB RAM, 128GB Storage, 4500mAh Battery', 'price': '₹54,999', 'price_range': 'Premium', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi 12 Pro', 'specifications': '6.73" Display, 8GB RAM, 256GB Storage, 4600mAh Battery', 'price': '₹39,999', 'price_range': 'Mid-range', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi Redmi Note 12 Pro+', 'specifications': '6.67" Display, 8GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹30,999', 'price_range': 'Mid-range', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi Redmi Note 12 Pro', 'specifications': '6.67" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹23,999', 'price_range': 'Budget', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi Redmi Note 12', 'specifications': '6.67" Display, 4GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹17,999', 'price_range': 'Budget', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi Redmi 12 5G', 'specifications': '6.79" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹13,999', 'price_range': 'Budget', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi Redmi 12C', 'specifications': '6.71" Display, 4GB RAM, 64GB Storage, 5000mAh Battery', 'price': '₹10,999', 'price_range': 'Budget', 'brand': 'Xiaomi', 'os': 'Android'},
            {'recommended_item': 'Xiaomi Redmi A2+', 'specifications': '6.52" Display, 3GB RAM, 64GB Storage, 5000mAh Battery', 'price': '₹8,999', 'price_range': 'Budget', 'brand': 'Xiaomi', 'os': 'Android'}
        ],
        'Poco': [
            {'recommended_item': 'Poco F5 Pro', 'specifications': '6.67" Display, 12GB RAM, 512GB Storage, 5160mAh Battery', 'price': '₹36,999', 'price_range': 'Mid-range', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco F5', 'specifications': '6.67" Display, 12GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹29,999', 'price_range': 'Mid-range', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco X5 Pro', 'specifications': '6.67" Display, 8GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹22,999', 'price_range': 'Budget', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco X5', 'specifications': '6.67" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹18,999', 'price_range': 'Budget', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco M5s', 'specifications': '6.43" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹14,999', 'price_range': 'Budget', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco M5', 'specifications': '6.58" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹12,999', 'price_range': 'Budget', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco C55', 'specifications': '6.71" Display, 4GB RAM, 64GB Storage, 5000mAh Battery', 'price': '₹9,999', 'price_range': 'Budget', 'brand': 'Poco', 'os': 'Android'},
            {'recommended_item': 'Poco C50', 'specifications': '6.52" Display, 3GB RAM, 32GB Storage, 5000mAh Battery', 'price': '₹7,999', 'price_range': 'Budget', 'brand': 'Poco', 'os': 'Android'}
        ],
        'Lenovo': [
            {'recommended_item': 'Lenovo Legion Phone Duel', 'specifications': '6.65" Display, 12GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹49,999', 'price_range': 'Premium', 'brand': 'Lenovo', 'os': 'Android'},
            {'recommended_item': 'Lenovo K13 Note', 'specifications': '6.5" Display, 4GB RAM, 64GB Storage, 5000mAh Battery', 'price': '₹9,999', 'price_range': 'Budget', 'brand': 'Lenovo', 'os': 'Android'},
            {'recommended_item': 'Lenovo A7', 'specifications': '6.09" Display, 2GB RAM, 32GB Storage, 4000mAh Battery', 'price': '₹6,999', 'price_range': 'Budget', 'brand': 'Lenovo', 'os': 'Android'}
        ],
        'Vivo': [
            {'recommended_item': 'Vivo X90 Pro', 'specifications': '6.78" Display, 12GB RAM, 256GB Storage, 4870mAh Battery', 'price': '₹84,999', 'price_range': 'Premium', 'brand': 'Vivo', 'os': 'Android'},
            {'recommended_item': 'Vivo V27 Pro', 'specifications': '6.78" Display, 12GB RAM, 256GB Storage, 4600mAh Battery', 'price': '₹37,999', 'price_range': 'Mid-range', 'brand': 'Vivo', 'os': 'Android'},
            {'recommended_item': 'Vivo V27', 'specifications': '6.78" Display, 8GB RAM, 128GB Storage, 4600mAh Battery', 'price': '₹32,999', 'price_range': 'Mid-range', 'brand': 'Vivo', 'os': 'Android'},
            {'recommended_item': 'Vivo T2 5G', 'specifications': '6.38" Display, 8GB RAM, 128GB Storage, 4500mAh Battery', 'price': '₹18,999', 'price_range': 'Budget', 'brand': 'Vivo', 'os': 'Android'},
            {'recommended_item': 'Vivo Y27 5G', 'specifications': '6.64" Display, 6GB RAM, 128GB Storage, 5000mAh Battery', 'price': '₹15,999', 'price_range': 'Budget', 'brand': 'Vivo', 'os': 'Android'}
        ],
        'Realme': [
            {'recommended_item': 'Realme GT 3', 'specifications': '6.74" Display, 16GB RAM, 1TB Storage, 4600mAh Battery', 'price': '₹56,999', 'price_range': 'Premium', 'brand': 'Realme', 'os': 'Android'},
            {'recommended_item': 'Realme 11 Pro+', 'specifications': '6.7" Display, 12GB RAM, 512GB Storage, 5000mAh Battery', 'price': '₹29,999', 'price_range': 'Mid-range', 'brand': 'Realme', 'os': 'Android'},
            {'recommended_item': 'Realme 11 Pro', 'specifications': '6.7" Display, 8GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹25,999', 'price_range': 'Mid-range', 'brand': 'Realme', 'os': 'Android'},
            {'recommended_item': 'Realme Narzo 60 Pro', 'specifications': '6.43" Display, 8GB RAM, 128GB Storage, 4300mAh Battery', 'price': '₹23,999', 'price_range': 'Budget', 'brand': 'Realme', 'os': 'Android'},
            {'recommended_item': 'Realme C55', 'specifications': '6.72" Display, 8GB RAM, 256GB Storage, 5000mAh Battery', 'price': '₹13,999', 'price_range': 'Budget', 'brand': 'Realme', 'os': 'Android'}
        ]
    }
    
    # If specific brand is selected, only show that brand's phones
    if brand and brand in all_recommendations:
        brand_phones = all_recommendations[brand]
        
        # Check OS compatibility first
        if os:
            # Apple only supports iOS
            if brand.lower() == 'apple' and os.lower() != 'ios':
                return []  # No recommendations for Apple + Android
            # All other brands only support Android
            elif brand.lower() != 'apple' and os.lower() != 'android':
                return []  # No recommendations for Android brands + iOS
        
        # Filter by budget and other criteria
        filtered = []
        for mobile in brand_phones:
            price_num = int(mobile['price'].replace('₹', '').replace(',', ''))
            
            if budget_min <= price_num <= budget_max:
                if not os or mobile['os'].lower() == os.lower():
                    filtered.append(mobile)
        
        return filtered[:10] if filtered else brand_phones[:8]
    
    # If no brand selected, show mixed recommendations
    mixed_recommendations = []
    for brand_name, phones in all_recommendations.items():
        for phone in phones[:2]:  # Take 2 phones from each brand
            price_num = int(phone['price'].replace('₹', '').replace(',', ''))
            if budget_min <= price_num <= budget_max:
                if not os or phone['os'].lower() == os.lower():
                    mixed_recommendations.append(phone)
    
    return mixed_recommendations[:10] if mixed_recommendations else list(all_recommendations['Samsung'])[:8]

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/')
def home():
    # Lazy load models on first request
    if not model_loaded:
        load_models()
    return render_template('index.html')

@lru_cache(maxsize=1000)
def cached_prediction(brand, os, processor, year, screen_size, storage, battery, ram):
    """Cache predictions for identical inputs"""
    input_data = pd.DataFrame([{
        'Brand': brand,
        'operating_system': os,
        'Processor': processor,
        'Release_year': year,
        'Screen-size': screen_size,
        'Internal_storage(GB)': storage,
        'Battery(mah)': battery,
        'RAM': ram
    }])
    
    if price_model:
        prediction = price_model.predict(input_data)
        return float(prediction[0])
    return 25000.0

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        
        # Use cached prediction
        predicted_price = cached_prediction(
            data['brand'],
            data['operating_system'], 
            data['processor'],
            int(data['release_year']),
            float(data['screen_size']),
            int(data['storage']),
            int(data['battery']),
            int(data['ram'])
        )
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@lru_cache(maxsize=500)
def cached_recommendations(budget_min, budget_max, brand, os, min_ram, min_storage, min_battery):
    """Cache recommendations for identical criteria"""
    return create_mobile_recommendations(
        budget_min, budget_max, brand, os, min_ram, min_storage, min_battery
    )

@app.route('/recommend', methods=['POST'])
def recommend_mobiles():
    try:
        data = request.get_json()
        budget_min = int(data.get('budget_min', 0))
        budget_max = int(data.get('budget_max', 100000))
        brand = data.get('brand', '')
        os = data.get('operating_system', '')
        min_ram = int(data.get('ram', 0))
        min_storage = int(data.get('storage', 0))
        min_battery = int(data.get('battery', 0))
        
        recommendations = cached_recommendations(
            budget_min, budget_max, brand, os, min_ram, min_storage, min_battery
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)