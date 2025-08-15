import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Cellphone Price Prediction")

# Inputs
sale = st.number_input("Sale (%)", min_value=0, max_value=100, value=10)
weight = st.number_input("Weight (g)", min_value=50.0, max_value=300.0, value=135.0)
resolution = st.number_input("Resolution (inches)", min_value=3.0, max_value=8.0, value=5.2)
ppi = st.number_input("PPI", min_value=100, max_value=600, value=424)
cpu_core = st.number_input("CPU Core Count", min_value=1, max_value=16, value=8)
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=5.0, value=1.35, format="%.2f")
internal_mem = st.number_input("Internal Memory (GB)", min_value=1, max_value=512, value=16)
ram = st.number_input("RAM (GB)", min_value=1, max_value=64, value=3)
rear_cam = st.number_input("Rear Camera (MP)", min_value=1, max_value=108, value=13)
front_cam = st.number_input("Front Camera (MP)", min_value=0, max_value=50, value=8)
battery = st.number_input("Battery (mAh)", min_value=1000, max_value=10000, value=2610)
thickness = st.number_input("Thickness (mm)", min_value=3.0, max_value=15.0, value=7.4)

# Prepare input array (must match training order!)
input_features = np.array([[sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem,
                            ram, rear_cam, front_cam, battery, thickness]])

# Scale inputs
input_scaled = scaler.transform(input_features)

if st.button("Predict Price"):
    pred_price = model.predict(input_scaled)[0]
    st.success(f"Predicted Price: Rs:{pred_price:.2f}")
