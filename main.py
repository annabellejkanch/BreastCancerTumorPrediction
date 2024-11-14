from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import os
import logging
import sys

app = Flask(__name__)

# Enhanced logging for Render deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# List of features for input
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
           'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
           'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
           'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

model = load_model('mlp_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    logger.info("Received prediction request")
    
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded")
        return render_template('index.html', 
                             error="Model not properly loaded. Please contact administrator.", 
                             features=features)
    
    try:
        # Log the form data
        logger.info(f"Form data received: {request.form}")
        
        # Extract and convert user input
        feature_values = []
        for feature in features:
            value = request.form.get(feature, '').strip()
            if not value:
                raise ValueError(f"Missing value for {feature}")
            feature_values.append(float(value))
        
        # Create DataFrame and log input
        input_df = pd.DataFrame([feature_values], columns=features)
        logger.info(f"Processing prediction for input shape: {input_df.shape}")
        
        # Preprocess the input
        features_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = float(prediction[0][0])
        diagnosis = 'Malignant' if probability > 0.5 else 'Benign'
        confidence = probability if probability > 0.5 else 1 - probability
        
        logger.info(f"Prediction complete: {diagnosis} with {confidence:.2%} confidence")
        
        return render_template('index.html',
                             prediction=diagnosis,
                             confidence=f"{confidence:.2%}",
                             features=features,
                             form_data=request.form)
                             
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return render_template('index.html',
                             error=f"Invalid input: {str(ve)}",
                             features=features,
                             form_data=request.form)
                             
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html',
                             error=f"An unexpected error occurred: {str(e)}",
                             features=features,
                             form_data=request.form)


if __name__ == '__main__':
    # Use PORT environment variable if available (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
