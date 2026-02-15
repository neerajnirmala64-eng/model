import os
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------
# Build absolute paths (CRITICAL FIX)
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# ------------------------------------------------
# Safe loading with clear error messages
# ------------------------------------------------
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error("model.pkl not found. Make sure it is committed to GitHub.")
        st.stop()

    if not os.path.exists(ENCODER_PATH):
        st.error("label_encoder.pkl not found. Make sure it is committed to GitHub.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder

model, encoder = load_assets()

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("Fuel Efficiency App")

car_name = st.text_input("Car Name")

horsepower = st.selectbox(
    "Horsepower",
    encoder["Horsepower"].classes_
)

displacement = st.selectbox(
    "Displacement",
    encoder["Displacement"].classes_
)

year = st.selectbox(
    "Model Year",
    encoder["Model year"].classes_
)

weight = st.number_input(
    "Weight of Vehicle",
    min_value=100,
    max_value=400
)

# ------------------------------------------------
# Prepare dataframe
# ------------------------------------------------
df = pd.DataFrame({
    "Car_name": [car_name],
    "Horsepower": [horsepower],
    "Displacement": [displacement],
    "Model year": [year],
    "Weight_of_vehicle": [weight]
})

# ------------------------------------------------
# Prediction button (MISSING IN YOUR CODE)
# ------------------------------------------------
if st.button("Predict Fuel Efficiency"):
    try:
        prediction = model.predict(df)
        st.success(f"Predicted Fuel Efficiency: {prediction[0]}")
    except Exception as e:
        st.error("Prediction failed. Check model & encoder compatibility.")
        st.exception(e)

