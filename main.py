from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import joblib

app = Flask(__name__)

# List of features for input
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
            'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
            'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
            'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Load model and scaler
model = load_model('mlp_model.h5')
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user input from the form based on feature names
        feature_values = [float(request.form.get(feature)) for feature in features]
        input_df = pd.DataFrame([feature_values], columns=features)
        print(input_df.shape)
        if len(input_df.shape) == 2:
            print("input_df is a 2D array.")
        else:
            print("input_df is not a 2D array.")        
        
        # Preprocess the input (scaling)
        features_scaled = scaler.transform([input_df])

        # Make prediction
        prediction = model.predict(features_scaled)
        diagnosis = 'Malignant' if prediction[0] > 0.5 else 'Benign'

        # Render the result back to the template
        return render_template('index.html', features=features, prediction=diagnosis)

    except Exception as e:
        # Handle any errors gracefully
        return render_template('index.html', features=features, error=str(e))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
