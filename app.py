import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('churn_model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_assets()

# --- 2. UI SETUP ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

if model is None or encoders is None:
    st.error("❌ Model files not found!")
    st.info("Please run 'python train_model.py' in your terminal first.")
    st.stop()

st.markdown("Predict if a customer will stay or leave based on their profile.")

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("Customer Details")

age = st.sidebar.slider('Age', 18, 100, 45)
tenure = st.sidebar.slider('Tenure (Months)', 0, 125, 12)
monthly = st.sidebar.number_input('Monthly Charges ($)', 30.0, 120.0, 75.0)
total = st.sidebar.number_input('Total Charges ($)', 0.0, 15000.0, 1000.0)

gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
contract = st.sidebar.selectbox('Contract Type', ('Month-to-Month', 'One-Year', 'Two-Year'))
internet = st.sidebar.selectbox('Internet Service', ('Fiber Optic', 'DSL', 'None'))
support = st.sidebar.selectbox('Tech Support', ('Yes', 'No'))

# --- 4. PREDICTION LOGIC ---
# Create input DataFrame
input_data = {
    'Age': age, 'Gender': gender, 'Tenure': tenure,
    'MonthlyCharges': monthly, 'ContractType': contract,
    'InternetService': internet, 'TotalCharges': total,
    'TechSupport': support
}
input_df = pd.DataFrame([input_data])

# Encode the input using the saved encoders
processed_df = input_df.copy()
for col, le in encoders.items():
    processed_df[col] = le.transform(input_df[col])

# Prediction Button
if st.button('Predict Churn Risk'):
    prediction = model.predict(processed_df)
    proba = model.predict_proba(processed_df)

    st.subheader('Result')
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: {proba[0][1]*100:.1f}% chance of churning.")
    else:
        st.success(f"✅ Low Risk: {proba[0][0]*100:.1f}% chance of staying.")
    
    st.divider()
    st.write("Customer Profile used for analysis:")
    st.dataframe(input_df)