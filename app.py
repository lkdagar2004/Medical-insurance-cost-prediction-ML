import streamlit as st
import pandas as pd
import numpy as np
import joblib

# st.cache_data load the model only once
@st.cache_data
def load_model_and_scaler():
    model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


model, scaler = load_model_and_scaler()
#User Interface
st.set_page_config(page_title="Insurance Charge Predictor", layout="wide")
st.title("👨‍⚕️ Medical Insurance Charge Predictor")
st.write("Enter your details on the left to get a prediction of your medical insurance charges.")

st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 60.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 10, 0)
is_smoker = st.sidebar.selectbox("Are you a smoker?", ("No", "Yes"))
is_female = st.sidebar.selectbox("Gender", ("Male", "Female"))
region_southeast = st.sidebar.selectbox("Are you from the Southeast region?", ("No", "Yes"))

def preprocess_input(age, bmi, children, is_smoker, is_female, region_southeast):

    smoker_val = 1 if is_smoker == "Yes" else 0
    female_val = 1 if is_female == "Female" else 0
    region_val = 1 if region_southeast == "Yes" else 0

    bmi_obese_val = 1 if bmi >= 30 else 0

    scaled_features = scaler.transform([[age, bmi, children]])

    feature_array = np.array([
        scaled_features[0, 0],
        scaled_features[0, 1],
        smoker_val,
        scaled_features[0, 2],
        region_val,
        female_val,
        bmi_obese_val
    ]).reshape(1, -1)

    return feature_array


# Predction
if st.sidebar.button("Predict Insurance Charge"):
    user_input = preprocess_input(age, bmi, children, is_smoker, is_female, region_southeast)

    prediction = model.predict(user_input)

    st.subheader("Predicted Insurance Charge")
    st.success(f"${prediction[0]:,.2f}")
    st.write("This prediction is trained Gradient Boosting model.")
