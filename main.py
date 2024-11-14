from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of features for input
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
           'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
           'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
           'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

def load_ml_components():
    """Load the model and scaler with error handling."""
    try:
        model = load_model('mlp_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading ML components: {str(e)}")
        raise RuntimeError("Failed to load machine learning components")

def validate_input(feature_values):
    """Validate input values."""
    if len(feature_values) != len(features):
        raise ValueError(f"Expected {len(features)} features, got {len(feature_values)}")
    
    for value in feature_values:
        if not isinstance(value, (int, float)):
            raise ValueError("All inputs must be numeric values")
        if value < 0:
            raise ValueError("All inputs must be positive values")

# Load model and scaler at startup
try:
    model, scaler = load_ml_components()
    logger.info("Successfully loaded model and scaler")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if model is None or scaler is None:
        return render_template('index.html', 
                             error="Model not properly loaded. Please contact administrator.", 
                             features=features)
    
    try:
        # Extract and convert user input
        feature_values = []
        for feature in features:
            value = request.form.get(feature, '').strip()
            if not value:
                raise ValueError(f"Missing value for {feature}")
            feature_values.append(float(value))

        # Validate input
        validate_input(feature_values)
        
        # Create DataFrame and log input
        input_df = pd.DataFrame([feature_values], columns=features)
        logger.info(f"Received input shape: {input_df.shape}")
        
        # Preprocess the input
        features_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = float(prediction[0][0])  # Convert to Python float
        diagnosis = 'Malignant' if probability > 0.5 else 'Benign'
        confidence = probability if probability > 0.5 else 1 - probability
        
        logger.info(f"Prediction made: {diagnosis} with {confidence:.2%} confidence")
        
        return render_template('index.html',
                             prediction=diagnosis,
                             confidence=f"{confidence:.2%}",
                             features=features)
                             
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return render_template('index.html',
                             error=f"Invalid input: {str(ve)}",
                             features=features)
                             
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html',
                             error="An unexpected error occurred. Please try again.",
                             features=features)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
