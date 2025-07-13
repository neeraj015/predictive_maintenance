# Build a Better Web Dashboard
#Upload new engine test data (e.g., CSV)


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown


# Theme selector dropdown
theme = st.sidebar.selectbox(" Choose Theme", ["Dark", "Light"])
# Define colors based on theme
if theme == "Dark":
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    accent_color = "#00D4B1"
else:
    bg_color = "#FFFFFF"
    text_color = "#262730"
    accent_color = "#F39C12"


st.sidebar.header(" Model Selector")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Linear Regression", "XGBoost"])

if model_choice == "Random Forest":
    #model = joblib.load("../models/rf_model.pkl")
    model_path = "rf_model.pkl"
    id_val = "17oS39_cXjtA6pKkjdJIaLJZO3taKyhTy"
elif model_choice == "Linear Regression":
    #model = joblib.load("../models/lr_model.pkl")
    model_path = "lr_model.pkl"
    id_val = "1FrKUhDsLTNvNo1aWpdZbHay7u36nNwDM"
else:
    #model = joblib.load("../models/xgb_model.pkl")
    model_path = "xgb_model.pkl"
    id_val = "1PNefuVJNFj4YgfTcXuVZbvlDvIdP39Je"



# Download from Google Drive if model doesn't exist
if not os.path.exists(model_path):
    gdown.download(id=id_val, output=model_path, quiet=False)

model = joblib.load(model_path)


#  Page config
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color};
    }}
    .css-1d391kg, .stMarkdown, .css-18e3th9 {{
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


#  Load model and scaler from parent directory
try:
    #model = joblib.load("../models/rf_model.pkl")
    if not os.path.exists(model_path):
        gdown.download(id=id_val, output=model_path, quiet=False)

    model = joblib.load(model_path)
    #scaler = joblib.load("../models/scaler.pkl")
    #with open("../models/feature_list.txt", "r") as f:
        #feature_list = [line.strip() for line in f.readlines()]
    # === Download scaler and feature list if not present ===
    if not os.path.exists("scaler.pkl"):
        gdown.download(id="1_jO5dGQltUXNBPPnusLvQIQS4842JMtL", output="scaler.pkl", quiet=False)

    if not os.path.exists("feature_list.txt"):
        gdown.download(id="1Um78HFNEkCrivBdTkAJkYNDCd8HsqkJy", output="feature_list.txt", quiet=False)

    # === Load them ===
    scaler = joblib.load("scaler.pkl")
    with open("feature_list.txt", "r") as f:
        feature_list = [line.strip() for line in f.readlines()]


except Exception as e:
    st.error(f" Failed to load model or scaler. Error: {e}")
    st.stop()

#  Page Header
st.title(" AI-Powered Predictive Maintenance - RUL Dashboard")
st.markdown("""
Upload sensor data from an engine to estimate its **Remaining Useful Life (RUL)** and visualize sensor trends.
""")

#  Sidebar file uploader
st.sidebar.header(" Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ✅ Main logic
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

        # Check for required features
        if all(feat in test_df.columns for feat in feature_list):

            # Normalize & predict
            test_scaled = scaler.transform(test_df[feature_list])
            test_df["Predicted_RUL"] = model.predict(test_scaled)

            # ✅ Use Tabs for layout
            tab1, tab2, tab3 = st.tabs([" Upload & Predict", " Sensor Trends", " Feature Importance"])

            with tab1:
                st.subheader(" Uploaded Data Preview")
                st.write(test_df.head())

                st.success(" RUL prediction completed!")
                st.subheader(" Predicted Remaining Useful Life (RUL)")
                st.write(test_df[["Predicted_RUL"]].head())

            with tab2:
                st.subheader(" Sensor Trends")
                sensor_cols = [col for col in test_df.columns if col.startswith("sensor_")]
                if sensor_cols:
                    selected_sensor = st.selectbox("Choose a sensor to visualize:", sensor_cols)
                    st.line_chart(test_df[selected_sensor])
                else:
                    st.warning(" No sensor columns found in uploaded file.")

            with tab3:
                st.subheader(" Feature Importance")
                importances = model.feature_importances_
                fi_df = pd.DataFrame({'Feature': feature_list, 'Importance': importances})
                fi_df = fi_df.sort_values(by='Importance', ascending=False)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
                st.pyplot(fig)

        else:
            st.error(" Uploaded file is missing one or more required input features.")
            st.markdown("###  Required features in your CSV:")
            for feat in feature_list:
                st.markdown(f"- `{feat}`")

    except Exception as e:
        st.error(f" Failed to process the uploaded file.\n\n**Error:** {e}")

else:
    # If no file uploaded, show dashboard info
    st.info(" Upload a `.csv` test file to start prediction.")
    st.markdown("---")
    st.subheader(" About This Dashboard")
    st.markdown("""
This tool predicts the **Remaining Useful Life (RUL)** of engines based on sensor data  
from the NASA Turbofan Engine dataset (CMAPSS). It's powered by a trained Random Forest model.

###  Instructions:
- Use the **left sidebar** to upload a CSV file of sensor readings.
- The app will normalize the data, run predictions, and show sensor trends.
- Input features must match the model's training features.

###  Expected CSV Format:
Include all required features such as:
- `op_setting_1`, `op_setting_2`, `op_setting_3`
- `sensor_2`, `sensor_3`, ..., `sensor_21` (excluding dropped constant sensors)
    """)

