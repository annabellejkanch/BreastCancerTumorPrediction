from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mlp_model.h5')

# StandardScaler used during model training (you should save and load this too)
scaler = StandardScaler()
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data from user input
    features = [float(request.form.get(feature)) for feature in ['radius_mean', 'texture_mean', 'perimeter_mean',
                                                                'area_mean', 'smoothness_mean', 'compactness_mean',
                                                                'concavity_mean', 'concave_points_mean', 'symmetry_mean',
                                                                'fractal_dimension_mean', 'radius_se', 'texture_se',
                                                                'perimeter_se', 'area_se', 'smoothness_se',
                                                                'compactness_se', 'concavity_se', 'concave_points_se',
                                                                'symmetry_se', 'fractal_dimension_se', 'radius_worst',
                                                                'texture_worst', 'perimeter_worst', 'area_worst',
                                                                'smoothness_worst', 'compactness_worst', 'concavity_worst',
                                                                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']]

    # Scale features using the same scaler as used in training
    features_scaled = scaler.fit_transform([features])

    # Make prediction using the model
    prediction = model.predict(features_scaled)
    
    # Convert prediction to "Malignant" or "Benign"
    result = "Malignant" if prediction[0] > 0.5 else "Benign"
    
    return render_template('index.html', prediction_text=f"The tumor is {result}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
