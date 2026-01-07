import pickle
import pandas as pd

# Load saved model
try:
    with open("Models/mobile_price_model (1).pkl", "rb") as file:
        model = pickle.load(file)
except:
    model = None

def calculate_realistic_price(brand, os, processor, year, screen_size, storage, battery, ram_gb):
    """Calculate realistic price based on specifications"""
    try:
        base_price = 8000
        
        brand_multipliers = {
            'Apple': 3.5, 'Samsung': 1.8, 'OnePlus': 2.2, 'Google': 2.5,
            'Xiaomi': 1.0, 'Poco': 0.9, 'Realme': 1.1, 'Oppo': 1.3,
            'Vivo': 1.2, 'Motorola': 1.1, 'Nokia': 1.0, 'Honor': 1.1,
            'Huawei': 1.4, 'Asus': 1.3, 'Sony': 1.6, 'LG': 1.2,
            'HTC': 1.3, 'Lenovo': 0.9, 'Other': 0.8
        }
        
        brand_mult = brand_multipliers.get(brand, 1.0)
        
        if storage <= 32: storage_mult = 0.8
        elif storage <= 64: storage_mult = 1.0
        elif storage <= 128: storage_mult = 1.3
        elif storage <= 256: storage_mult = 1.7
        else: storage_mult = 2.2
        
        if ram_gb <= 3: ram_mult = 0.8
        elif ram_gb <= 4: ram_mult = 1.0
        elif ram_gb <= 6: ram_mult = 1.2
        elif ram_gb <= 8: ram_mult = 1.4
        elif ram_gb <= 12: ram_mult = 1.7
        else: ram_mult = 2.0
        
        year_diff = int(year) - 2024
        if year_diff >= 0:
            year_mult = 1.0 + (year_diff * 0.1)
        else:
            year_mult = max(0.5, 1.0 + (year_diff * 0.15))
        
        battery_mult = 1.1 if battery >= 5000 else (1.0 if battery >= 4000 else 0.95)
        screen_mult = 1.05 if screen_size >= 6.5 else (1.0 if screen_size >= 6.0 else 0.95)
        os_mult = 1.2 if os.lower() == 'ios' else 1.0
        
        calculated_price = (base_price * brand_mult * storage_mult * ram_mult * 
                          year_mult * battery_mult * screen_mult * os_mult)
        
        return max(min(calculated_price, 150000), 5000)
    except:
        return 15000.0

def predict_price(input_data):
    ram_gb = input_data.get('RAM', 6)
    if ram_gb > 100:
        ram_gb = ram_gb / 1024
    
    if model:
        try:
            df = pd.DataFrame([{
                'Brand': input_data.get('Brand', 'Xiaomi'),
                'operating_system': input_data.get('operating_system', 'Android'),
                'Processor': input_data.get('Processor', 'octa-core'),
                'Release_year': int(input_data.get('Release_year', 2022)),
                'Screen-size': float(input_data.get('Screen-size', 6.5)),
                'Internal_storage(GB)': int(input_data.get('Internal_storage(GB)', 128)),
                'Battery(mah)': int(input_data.get('Battery(mah)', 4000)),
                'RAM': ram_gb
            }])
            
            prediction = model.predict(df)
            predicted_price = float(prediction[0])
            
            if predicted_price > 35000 or predicted_price < 3000:
                return calculate_realistic_price(
                    input_data.get('Brand', 'Xiaomi'),
                    input_data.get('operating_system', 'Android'),
                    input_data.get('Processor', 'octa-core'),
                    input_data.get('Release_year', 2022),
                    input_data.get('Screen-size', 6.5),
                    input_data.get('Internal_storage(GB)', 128),
                    input_data.get('Battery(mah)', 4000),
                    ram_gb
                )
            
            return max(predicted_price, 5000)
        except:
            pass
    
    return calculate_realistic_price(
        input_data.get('Brand', 'Xiaomi'),
        input_data.get('operating_system', 'Android'),
        input_data.get('Processor', 'octa-core'),
        input_data.get('Release_year', 2022),
        input_data.get('Screen-size', 6.5),
        input_data.get('Internal_storage(GB)', 128),
        input_data.get('Battery(mah)', 4000),
        ram_gb
    )

# Example test
if __name__ == "__main__":
    sample = {
        "Brand": "Samsung",
        "operating_system": "Android",
        "Processor": "octa-core",
        "Release_year": 2023,
        "Screen-size": 6.8,
        "Internal_storage(GB)": 128,
        "Battery(mah)": 5000,
        "RAM": 8
    }
    predicted = predict_price(sample)
    print(f"Predicted Price: ₹{predicted:,.0f}")
    
    # Test with different specs
    apple_sample = {
        "Brand": "Apple",
        "operating_system": "iOS",
        "Processor": "hexa-core",
        "Release_year": 2022,
        "Screen-size": 6.1,
        "Internal_storage(GB)": 128,
        "Battery(mah)": 3279,
        "RAM": 6
    }
    apple_predicted = predict_price(apple_sample)
    print(f"Apple Predicted Price: ₹{apple_predicted:,.0f}")
