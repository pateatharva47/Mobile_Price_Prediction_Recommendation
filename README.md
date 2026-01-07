# 📱 Mobile Price Prediction & Recommendation System

This project predicts the **price of a mobile phone** based on its specifications and also provides **mobile recommendations** based on user budget and preferences using machine learning techniques.

The application is built using **Python and Flask** with a simple and user-friendly web interface.

---

## 🚀 Tech Stack
- Python 3.10+
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- HTML, CSS, JavaScript (Frontend)
- VS Code (Development)
- Render (Deployment)

---

## 🧠 Problem Statement
Choosing the right mobile phone is difficult due to:
- Large number of brands
- Different specifications
- Wide price range

This project helps users by:
- Predicting mobile price
- Recommending best mobiles within a given budget

---

## 📊 Features of the Project

### 🔹 Mobile Price Prediction
Predicts the price of a mobile phone based on:
- Brand  
- Operating System  
- RAM  
- Storage  
- Battery capacity  
- Screen size  
- Processor type  

### 🔹 Mobile Recommendation System
Recommends mobiles based on:
- Budget range  
- Brand preference  
- OS preference  
- RAM, Storage, and Battery requirements  

Uses **FP-Growth association rules** along with fallback logic for recommendations.

---

## 🏗️ Project Structure

```bash
Mobile_Price_Prediction_Recommendation/
│── app.py # Main Flask application
│── run.py # Application run file
│── Procfile # Deployment configuration
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│
├── Models/
│ ├── mobile_price_model.pkl # Trained ML model
│ └── fpgrowth_rules.pkl # Recommendation rules
│
├── Scripts/
│ ├── predict_price.py # Price prediction logic
│ └── recommendation.py # Recommendation logic
│
├── data/
│ ├── mobilesdf.csv # Dataset
│ └── cleaned_data.csv
│
├── notebooks/
│ ├── Mobile_Price_prediction.ipynb
│ └── FP_Growth.ipynb
│
├── templates/
│ ├── base.html # Common layout template
│ └── index.html # Web UI
│
└── static/
├── css/
└── js/
```

---

## 📦 Installation & Running the Project

### ✅ Step 1: Clone the Repository
```bash
git clone https://github.com/pateatharva47/Mobile_Price_Prediction_Recommendation.git
cd Mobile_Price_Prediction_Recommendation
```

### ✅ Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### ✅ Step 3: Run the Flask Application
```bash
python run.py
```


Then open your browser and go to: http://127.0.0.1:5000/


---

## 🌐 Deployment

The project can be deployed on **Render** using the following configuration:

- Python environment
- Start command: `python run.py`
- Public GitHub repository

---

## 🧠 Model Information

- Algorithm: Random Forest Regressor
- Model Accuracy: ~88%
- Recommendation Technique: FP-Growth

### Output:
- Predicted mobile price
- Recommended mobile phones

---

## 📌 Key Points

- Machine learning model is pre-trained
- Model is loaded during runtime (no training on server)
- Simple UI and easy to use
- Suitable for academic projects and demonstrations

---

## ✅ Submission Notes

- Includes trained model files
- No external data download required
- Ready for demo and deployment
- Written in simple and understandable way


