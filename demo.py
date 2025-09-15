import streamlit as st
import joblib
import numpy as np
from os import path

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Flower Classification App",
    page_icon="🌸",
    layout="centered"
)

# -------------------------------
# Welcome Image (from web)
# -------------------------------
st.image("LsBKyoS.png", use_container_width=True)

# -------------------------------
# Custom CSS for styling
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f9f9f9, #ffe6f0);
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1c74;
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 15px;
        border-radius: 12px;
        background: white;
        border: 2px solid #ff4b4b;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title & description
# -------------------------------
st.title("🌸 Flower Classification App")
st.markdown("Enter the flower measurements below to predict the species.")

# -------------------------------
# Load trained model
# -------------------------------
file_name = "Iris_classifier.pkl"
model = joblib.load(path.join("model", file_name))

# -------------------------------
# Input fields
# -------------------------------
sl = st.number_input("🌱 Sepal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sw = st.number_input("🌱 Sepal width (cm)", min_value=0.0, max_value=10.0, step=0.1)
pl = st.number_input("🌸 Petal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
pw = st.number_input("🌸 Petal width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔮 Predict Flower"):
    pred = model.predict([[sl, sw, pl, pw]])
    classes = ["Setosa", "Versicolor", "Virginica"]
    flower = classes[pred[0]]

    st.markdown(
        f"<div class='prediction-box'>🌺 The flower is: <span style='color:#ff1c74'>{flower}</span></div>",
        unsafe_allow_html=True
    )
