import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --- 1. Load the Saved Model and Scaler ---
# Use st.cache_data to load the model only once
@st.cache_data
def load_model_and_scaler():
    model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


model, scaler = load_model_and_scaler()

# --- 2. Create the User Interface ---
st.set_page_config(page_title="Insurance Charge Predictor", layout="wide")
st.title("👨‍⚕️ Medical Insurance Charge Predictor")
st.write("Enter your details on the left to get a prediction of your medical insurance charges.")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Create input fields in the sidebar
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 60.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 10, 0)
is_smoker = st.sidebar.selectbox("Are you a smoker?", ("No", "Yes"))
is_female = st.sidebar.selectbox("Gender", ("Male", "Female"))
region_southeast = st.sidebar.selectbox("Are you from the Southeast region?", ("No", "Yes"))


# --- 3. Preprocess User Input for Prediction ---
def preprocess_input(age, bmi, children, is_smoker, is_female, region_southeast):
    # Convert categorical inputs to numerical format
    smoker_val = 1 if is_smoker == "Yes" else 0
    female_val = 1 if is_female == "Female" else 0
    region_val = 1 if region_southeast == "Yes" else 0

    # Determine bmi_categories_obese from BMI
    bmi_obese_val = 1 if bmi >= 30 else 0

    # Scale the numerical features using the saved scaler
    scaled_features = scaler.transform([[age, bmi, children]])

    # Create the final feature array in the correct order
    # This order MUST match the order of columns the model was trained on
    feature_array = np.array([
        scaled_features[0, 0],  # scaled age
        scaled_features[0, 1],  # scaled bmi
        smoker_val,
        scaled_features[0, 2],  # scaled children
        region_val,
        female_val,
        bmi_obese_val
    ]).reshape(1, -1)

    return feature_array


# --- 4. Make a Prediction ---
if st.sidebar.button("Predict Insurance Charge"):
    # Preprocess the input
    user_input = preprocess_input(age, bmi, children, is_smoker, is_female, region_southeast)

    # Make the prediction
    prediction = model.predict(user_input)

    # Display the result
    st.subheader("Predicted Insurance Charge")
    st.success(f"${prediction[0]:,.2f}")
    st.write("This prediction is based on the details you provided and a trained Gradient Boosting model.")
