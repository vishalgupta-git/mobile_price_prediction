import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('best_rf.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Cellphone Price Prediction")

st.image("img/mobile.png", caption="Cellphone Image")

# Inputs
sale = st.slider("🛍️ Adjust Sale Percentage", min_value=0, max_value=100, value=10, step=1)

weight_option = st.radio("📦 Choose Weight Input Method", ["Quick Select", "Custom"])
if weight_option == "Quick Select":
    weight = st.selectbox("Select Weight (g)", options=[100, 120, 135, 150, 180, 200, 250, 300], index=2)
else:
    weight = st.slider("Adjust Weight (g)", min_value=50.0, max_value=300.0, value=135.0, step=0.25)

resolution = st.slider("📱 Screen Size (inches)", min_value=3.0, max_value=10.0, value=5.0)

ppi = st.selectbox("🖼️ Pixels Per Inch (PPI)", options=[160, 240, 300, 326, 400, 424, 500, 600], index=5)

cpu_core = st.radio("🧠 CPU Core Count", options=[2, 4, 6, 8, 12, 16], index=3)

cpu_freq = st.slider("⚡ CPU Frequency (GHz)", min_value=0.5, max_value=5.0, value=1.35, step=0.05)

internal_mem = st.selectbox("💾 Internal Memory (GB)", options=[8, 16, 32, 64, 128, 256, 512], index=1)

ram = st.selectbox("📈 RAM (GB)", options=[2, 3, 4, 6, 8, 12, 16, 32, 64], index=1)

rear_cam = st.slider("📷 Rear Camera (MP)", min_value=1, max_value=108, value=13)

front_cam = st.slider("🤳 Front Camera (MP)", min_value=0, max_value=50, value=8)

battery = st.slider("🔋 Battery Capacity (mAh)", min_value=1000, max_value=10000, value=2610, step=100)

thickness = st.number_input("📏 Thickness (mm)", min_value=3.0, max_value=15.0, value=7.4, step=0.1)

# Prepare input data dictionary (column names must match training data)
input_dict = {
    "Sale": sale,
    "weight": weight,
    "resoloution": resolution,
    "ppi": ppi,
    "cpu core": cpu_core,
    "cpu freq": cpu_freq,
    "internal mem": internal_mem,
    "ram": ram,
    "RearCam": rear_cam,
    "Front_Cam": front_cam,
    "battery": battery,
    "thickness": thickness
}

# Create DataFrame from input dict (1 row)
input_df = pd.DataFrame([input_dict])

# Scale inputs using scaler
input_scaled_array = scaler.transform(input_df)
input_scaled = pd.DataFrame(input_scaled_array, columns=input_df.columns)

if st.button("Predict Price"):
    st.write("### Input Summary")
    st.json(input_dict)

    pred_price = model.predict(input_scaled)[0]
    st.balloons()
    st.success(f"Predicted Price: Rs: {pred_price:.2f}")
