import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD ASSETS ---
model = joblib.load('churn_model.pkl')
encoders = joblib.load('encoders.pkl')

# --- UI SETUP ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("""
Predict if a customer will stay or leave based on their profile. 
Fill in the details below to see the result.
""")

st.sidebar.header("Customer Input Features")

def user_input_features():
    # Numerical Inputs
    age = st.sidebar.slider('Age', 18, 100, 45)
    tenure = st.sidebar.slider('Tenure (Months)', 0, 125, 12)
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', 30.0, 120.0, 75.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', 0.0, 15000.0, 1000.0)
    
    # Categorical Inputs
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    contract = st.sidebar.selectbox('Contract Type', ('Month-to-Month', 'One-Year', 'Two-Year'))
    internet = st.sidebar.selectbox('Internet Service', ('Fiber Optic', 'DSL', 'None'))
    support = st.sidebar.selectbox('Tech Support', ('Yes', 'No'))

    # Dictionary for conversion
    data = {
        'Age': age,
        'Gender': gender,
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'ContractType': contract,
        'InternetService': internet,
        'TotalCharges': total_charges,
        'TechSupport': support
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- PREPROCESSING ---
# We must encode the inputs exactly how we did during training
processed_df = input_df.copy()

for col, le in encoders.items():
    processed_df[col] = le.transform(input_df[col])

# --- PREDICTION ---
if st.button('Predict Churn'):
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('⚠️ High Risk: This customer is likely to CHURN.')
    else:
        st.success('✅ Low Risk: This customer is likely to STAY.')

    # Probability metrics
    st.write(f"**Confidence Level:** {np.max(prediction_proba)*100:.2f}%")
    
    st.divider()
    st.write("Current Input Values:")
    st.dataframe(input_df)