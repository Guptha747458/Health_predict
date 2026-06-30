import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ── Helper to resolve paths relative to this script's directory ─────────────
# This ensures joblib files are found correctly on Streamlit Cloud,
# where the current working directory may differ from the script's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


# ── Load model and pipeline once, cached across reruns ──────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load(_path('random_forest_model.joblib'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load(_path('preprocessing_pipeline.joblib'))
        return pipeline
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        st.stop()


loaded_rf_model = load_model()
preprocessing_pipeline = load_pipeline()


# ── Prediction function ──────────────────────────────────────────────────────
def predict_risk_level(data: pd.DataFrame):
    try:
        data_processed = preprocessing_pipeline.transform(data)
        prediction = loaded_rf_model.predict(data_processed)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


# ── App UI ───────────────────────────────────────────────────────────────────
st.title("Health Risk Prediction App")
st.write("Enter the patient's vital signs and other information to predict their health risk level.")

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=100, value=20, step=1)
    oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=95, step=1)
    o2_scale = st.selectbox("O2 Scale", [1, 2])
    temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)

with col2:
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=300, value=120, step=1)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=80, step=1)
    consciousness = st.selectbox(
        "Consciousness Level",
        ['A', 'V', 'P', 'U', 'C'],
        help="A=Alert, V=Voice, P=Pain, U=Unresponsive, C=Confused"
    )
    on_oxygen = st.selectbox("On Supplemental Oxygen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')


# ── Build input DataFrame ────────────────────────────────────────────────────
input_data = pd.DataFrame({
    'Respiratory_Rate': [int(respiratory_rate)],
    'Oxygen_Saturation': [int(oxygen_saturation)],
    'O2_Scale': [int(o2_scale)],
    'Systolic_BP': [int(systolic_bp)],
    'Heart_Rate': [int(heart_rate)],
    'Temperature': [float(temperature)],
    'Consciousness': [consciousness],
    'On_Oxygen': [int(on_oxygen)]
})


# ── Predict button ───────────────────────────────────────────────────────────
if st.button("Predict Risk Level", type="primary"):
    with st.spinner("Analyzing..."):
        predicted_risk = predict_risk_level(input_data)

    if predicted_risk is not None:
        st.subheader("Predicted Health Risk Level:")

        if predicted_risk == 'High':
            st.error(f"🚨 Risk Level: **{predicted_risk}**")
        elif predicted_risk == 'Medium':
            st.warning(f"⚠️ Risk Level: **{predicted_risk}**")
        elif predicted_risk == 'Low':
            st.info(f"ℹ️ Risk Level: **{predicted_risk}**")
        elif predicted_risk == 'Normal':
            st.success(f"✅ Risk Level: **{predicted_risk}**")
        else:
            st.write(f"Risk Level: **{predicted_risk}**")