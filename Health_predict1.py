# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("trained_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        data["Respiratory_Rate"],
        data["Oxygen_Saturation"],
        data["Systolic_BP"],
        data["Heart_Rate"],
        data["Temperature"]
    ]])
    prediction = model.predict(features)
    return jsonify({"Predicted Risk Level": str(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)