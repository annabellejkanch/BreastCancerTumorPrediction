from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

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
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]
        features_df = pd.DataFrame([features], columns = features)
        scaled = scaler.transform(features_df)  
        
        # Make prediction
        prediction = model.predict(scaled)
        
        # Convert prediction to string
        output = str(prediction[0])
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Tumor Type: {output}')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    # Use PORT environment variable if available (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
