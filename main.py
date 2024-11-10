from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mlp_model.h5')

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# List of feature names in the same order as used in the model
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
            'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
            'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
            'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data from user input
        input_data = []
        for feature in features:
            value = request.form.get(feature)
            if value:
                try:
                    input_data.append(float(value))  # Convert to float
                except ValueError:
                    return render_template('index.html', prediction_text=f"Invalid input for {feature}. Please enter numeric values.", features=features)
            else:
                return render_template('index.html', prediction_text="All fields must be filled in.", features=features)

        # Impute missing values in the input data using the same imputer as used in training
        input_data_imputed = imputer.transform([input_data])

        # Scale the input data using the scaler
        input_data_scaled = scaler.transform(input_data_imputed)

        # Make the prediction using the trained model
        prediction = model.predict(input_data_scaled)

        # Convert the prediction to "Malignant" or "Benign"
        result = "Malignant" if prediction[0] > 0.5 else "Benign"

        return render_template('index.html', prediction_text=f"The tumor is {result}", features=features)

    except Exception as e:
        print("Error:", str(e))  # Log error message for debugging
        return render_template('index.html', prediction_text="Error in input data. Please check the values and try again.", features=features)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
