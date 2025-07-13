import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scaler, and feature list
model = joblib.load('../models/rf_model.pkl')

try:
    scaler = joblib.load('../models/scaler.pkl')  # optional
except:
    scaler = None

with open('../models/feature_list.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

# Streamlit app layout
st.set_page_config(page_title="Predictive Maintenance - RUL Predictor", layout="centered")
st.title("🔧 Predictive Maintenance")
st.markdown("Enter engine sensor values to predict Remaining Useful Life (RUL).")

# Input form
user_input = {}
for feat in features:
    user_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")

# Predict button
if st.button("🔮 Predict RUL"):
    input_df = pd.DataFrame([user_input])

    # Apply scaler if it exists
    if scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df

    prediction = model.predict(input_scaled)[0]
    st.success(f" Predicted RUL: {prediction:.2f} cycles")

    st.markdown("---")
    st.subheader("📊 Your Input")
    st.dataframe(input_df.T, use_container_width=True)



'''
#initial code
import streamlit as st

st.title("AI-Powered Predictive Maintenance")
st.write("This is a test dashboard.")'''

# in ternimal use 
#cd app
#streamlit run app.py

