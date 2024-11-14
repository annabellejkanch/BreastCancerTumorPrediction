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

def ValuePredictor(to_predict_list):
    # Convert the input list to a numpy array and reshape it to (1, 30)
    to_predict = pd.DataFrame([to_predict_list], columns=features)
    
    # Scale the input data
    scaled_data = scaler.transform(to_predict)
    
    # Make prediction using the model
    result = model.predict(scaled_data)
    
    return result[0][0]

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data, make predictions, etc.
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))  # Ensure the input is floats

        # Use the ValuePredictor function to get the prediction
        result = ValuePredictor(to_predict_list)
        
        # Interpret the result (adjust according to your use case)
        if result == 1:
            prediction = 'Malignant Tumor'
        else:
            prediction = 'Benign Tumor'
        
        return render_template("predict.html", prediction=prediction)
    
    # Handle GET request to just show the form
    return render_template("index.html", features=features)



#if __name__ == '__main__':
    # Use PORT environment variable if available (for Render)
#    port = int(os.environ.get('PORT', 5000))       
#    app.run(host='0.0.0.0', port=port)
