import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the saved model
try:
    loaded_rf_model = joblib.load('random_forest_model.joblib')
except FileNotFoundError:
    st.error("Error: Model file 'random_forest_model.joblib' not found. Please train and save the model first.")
    st.stop()

# Load the fitted preprocessing pipeline
try:
    preprocessing_pipeline = joblib.load('preprocessing_pipeline.joblib')
except FileNotFoundError:
    st.error("Error: Preprocessing pipeline file 'preprocessing_pipeline.joblib' not found. Please train and save the pipeline first.")
    st.stop()

# Function to preprocess new data and make prediction
def predict_risk_level(data):
    # Ensure columns are in the correct order and handle potential missing columns in new data if necessary
    # For simplicity, this assumes the input data DataFrame has the required columns in a reasonable order
    # In a real app, you might add more robust column handling

    try:
        # Apply preprocessing
        data_processed = preprocessing_pipeline.transform(data)

        # Make prediction
        prediction = loaded_rf_model.predict(data_processed)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction preprocessing: {e}")
        return None


# Streamlit App Title and Description
st.title("Health Risk Prediction App")
st.write("Enter the patient's vital signs and other information to predict their health risk level.")

# Input fields for user
st.header("Patient Information")

respiratory_rate = st.number_input("Respiratory Rate", min_value=0, value=20)
oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=95)
o2_scale = st.selectbox("O2 Scale", [1, 2])
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=120)
heart_rate = st.number_input("Heart Rate", min_value=0, value=80)
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=37.0)
consciousness = st.selectbox("Consciousness Level", ['A', 'V', 'P', 'U', 'C'])
on_oxygen = st.selectbox("On Oxygen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')


# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Respiratory_Rate': [respiratory_rate],
    'Oxygen_Saturation': [oxygen_saturation],
    'O2_Scale': [o2_scale],
    'Systolic_BP': [systolic_bp],
    'Heart_Rate': [heart_rate],
    'Temperature': [temperature],
    'Consciousness': [consciousness],
    'On_Oxygen': [on_oxygen]
})


# Predict button
if st.button("Predict Risk Level"):
    predicted_risk = predict_risk_level(input_data)

    if predicted_risk:
        st.subheader("Predicted Health Risk Level:")
        st.write(f"**{predicted_risk}**")