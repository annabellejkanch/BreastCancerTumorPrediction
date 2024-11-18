from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# List of features for input
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
           'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
           'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
           'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Load model and scaler
model = load_model('mlp_model.h5')
scaler = joblib.load('scaler.pkl')

def ValuePredictor(to_predict_list):
    try:
        # Convert the input list to a DataFrame
        to_predict = pd.DataFrame([to_predict_list], columns=features)
        print("Input data shape:", to_predict.shape)  # Debug print
        
        # Scale the input data
        scaled_data = scaler.transform(to_predict)
        print("Scaled data shape:", scaled_data.shape)  # Debug print
        
        # Make prediction using the model
        result = model.predict(scaled_data)
        print("Prediction result:", result)  # Debug print
        
        return result
    except Exception as e:
        print(f"Error in ValuePredictor: {str(e)}")
        return None

@app.route("/", methods=['GET', 'POST'])
def home():
    prediction = None
    
    try:
        if request.method == 'POST':
            print("Received POST request")  # Debug print
            print("Form data:", request.form)  # Debug print
            
            # Get the form data and convert to list of floats
            to_predict_list = []
            for feature in features:
                value = request.form.get(feature)
                print(f"Feature {feature}: {value}")  # Debug print
                to_predict_list.append(float(value))
            
            print("Processed input list:", to_predict_list)  # Debug print
            
            # Get prediction
            result = ValuePredictor(to_predict_list)
            
            if result is not None:
                # Interpret the result
                print("Raw prediction:", result)  # Debug print
                prediction = 'Malignant Tumor' if result[0][0] == 1 else 'Benign Tumor'
                print("Final prediction:", prediction)  # Debug print
            else:
                prediction = "Error: Could not make prediction"
                
    except Exception as e:
        print(f"Error in home route: {str(e)}")
        prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction, features=features)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
