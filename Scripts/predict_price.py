import pickle
import pandas as pd

# Load saved model
with open("mobile_price_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_price(input_data):
    # input_data is a dict → convert to DataFrame
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return float(prediction[0])

# Example test
if __name__ == "__main__":
    sample = {
        "Brand": "Samsung",
        "operating_system": "Android",
        "Processor": "Snapdragon",
        "Release_year": 2023,
        "Screen-size": 6.8,
        "Internal_storage(GB)": 128,
        "Battery(mah)": 5000,
        "RAM": 8
    }
    print("Predicted Price:", predict_price(sample))
