from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

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

@app.route("/")
def home():
    return render_template("temp.html")
  
@app.route("/predict", methods=["POST"])
def predict:
    float_features = [float(x) for x in request.form.values()]
    to_predict = pd.DataFrame([float_features], columns = features)
    scaled_data = scaler.transform(to_predict)
    result = model.predict(scaled_data)
    if result == 1:
      result = "Malginant"
    else"
      result = "Benign"
    return render_template("temp.html", prediction_text = "The predicted tumor is {}".format(result))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
