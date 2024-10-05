import pickle
from django.shortcuts import render
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the Model and Scalar
regmodel = pickle.load(open(r'C:\Users\Dell\Desktop\endtoendmlproject\californiahousepricing\regmodel.pkl', 'rb'))
scalar = pickle.load(open(r'C:\Users\Dell\Desktop\endtoendmlproject\californiahousepricing\scaling.pkl', 'rb'))  # Assuming `scaling.pkl` is the scaler file

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))  # Ensure scalar is correctly defined
    output = regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])  # Returning the predicted value as JSON response

if __name__ == "__main__":
    app.run(debug=True)
