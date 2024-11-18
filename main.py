from flask import Flask, render_template, request
import numpy as pd
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
    # Convert the input list to a DataFrame
    to_predict = pd.DataFrame([to_predict_list], columns=features)
    
    # Scale the input data
    scaled_data = scaler.transform(to_predict)
    
    # Make prediction using the model
    result = model.predict(scaled_data)
    
    return result

@app.route("/", methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get the form data and convert to list of floats
            to_predict_list = list(request.form.values())
            to_predict_list = list(map(float, to_predict_list))
            
            # Get prediction
            result = ValuePredictor(to_predict_list)
            
            # Interpret the result
            prediction = 'Malignant Tumor' if result[0][0] == 1 else 'Benign Tumor'
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction, features=features)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
