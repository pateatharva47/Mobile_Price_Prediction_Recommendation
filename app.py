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

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)

price_model = None
recommendation_rules = None
mobile_database = None

def create_smart_model():
    class SmartMobilePriceModel:
        def __init__(self):
            self.brand_base_prices = {
                'Apple': 45000, 'Samsung': 25000, 'OnePlus': 30000, 'Google': 40000,
                'Xiaomi': 15000, 'Realme': 12000, 'Vivo': 18000, 'Oppo': 20000,
                'Huawei': 25000, 'Nokia': 10000, 'Poco': 12000, 'Motorola': 15000,
                'Honor': 16000, 'iQOO': 22000, 'Asus': 28000, 'LG': 20000
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
    global price_model, recommendation_rules, mobile_database
    try:
        print("Loading models...")
        data_path = "data/cleaned_data1 (1).csv"
        if os.path.exists(data_path):
            print("Loading training data...")
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
            print("Using fallback model")
            price_model = create_smart_model()
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
        return price_model, recommendation_rules
    except Exception as e:
        print(f"Error loading models: {e}")
        price_model = create_smart_model()
        return price_model, None

def create_mobile_recommendations(budget_min=0, budget_max=100000, brand='', os='', min_ram=0, min_storage=0, min_battery=0):
    global mobile_database
    
    if mobile_database is not None:
        df = mobile_database.copy()
        
        # Filter by budget
        df = df[(df['Price_in_India'] >= budget_min) & (df['Price_in_India'] <= budget_max)]
        
        # Filter by brand if specified
        if brand:
            df = df[df['Brand'].str.lower() == brand.lower()]
        
        # Filter by OS if specified
        if os:
            df = df[df['operating_system'].str.lower() == os.lower()]
        
        # Filter by RAM if specified
        if min_ram > 0:
            df = df[df['RAM'] >= min_ram * 1024]  # Convert GB to MB
        
        # Filter by storage if specified
        if min_storage > 0:
            df = df[df['Internal_storage(GB)'] >= min_storage]
        
        # Filter by battery if specified
        if min_battery > 0:
            df = df[df['Battery(mah)'] >= min_battery]
        
        # Sort by price and get top 5
        df = df.sort_values('Price_in_India').head(5)
        
        recommendations = []
        for _, row in df.iterrows():
            recommendations.append({
                'recommended_item': f"{row['Brand']} {row['operating_system']} Phone",
                'specifications': f"{row['Screen-size']}\" Display, {int(row['RAM']/1024)}GB RAM, {int(row['Internal_storage(GB)'])}GB Storage, {int(row['Battery(mah)'])}mAh Battery",
                'price': f"₹{int(row['Price_in_India']):,}",
                'price_range': 'Budget' if row['Price_in_India'] < 25000 else 'Mid-range' if row['Price_in_India'] < 50000 else 'Premium',
                'brand': row['Brand'],
                'os': row['operating_system']
            })
        
        return recommendations
    
    # Fallback recommendations
    recommendations = [
        {
            'recommended_item': 'Samsung Galaxy A54 5G',
            'specifications': '6.4" Display, 8GB RAM, 128GB Storage, 5000mAh Battery',
            'price': '₹38,999',
            'price_range': 'Mid-range',
            'brand': 'Samsung',
            'os': 'Android'
        },
        {
            'recommended_item': 'Xiaomi Redmi Note 12 Pro',
            'specifications': '6.67" Display, 6GB RAM, 128GB Storage, 5000mAh Battery',
            'price': '₹23,999',
            'price_range': 'Budget',
            'brand': 'Xiaomi',
            'os': 'Android'
        },
        {
            'recommended_item': 'OnePlus Nord CE 3 Lite',
            'specifications': '6.72" Display, 8GB RAM, 128GB Storage, 5000mAh Battery',
            'price': '₹19,999',
            'price_range': 'Budget',
            'brand': 'OnePlus',
            'os': 'Android'
        }
    ]
    
    # Filter fallback recommendations
    filtered = []
    for mobile in recommendations:
        price_num = int(mobile['price'].replace('₹', '').replace(',', ''))
        if budget_min <= price_num <= budget_max:
            if not brand or mobile['brand'].lower() == brand.lower():
                if not os or mobile['os'].lower() == os.lower():
                    filtered.append(mobile)
    
    return filtered[:5]

load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([{
            'Brand': data['brand'],
            'operating_system': data['operating_system'],
            'Processor': data['processor'],
            'Release_year': int(data['release_year']),
            'Screen-size': float(data['screen_size']),
            'Internal_storage(GB)': int(data['storage']),
            'Battery(mah)': int(data['battery']),
            'RAM': int(data['ram'])
        }])
        
        if price_model:
            prediction = price_model.predict(input_data)
            predicted_price = float(prediction[0])
        else:
            predicted_price = 25000
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

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
        
        recommendations = create_mobile_recommendations(
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