import streamlit as st
import pandas as pd
import numpy as np
import joblib

# page title and layout
st.title("Credit Card Fraud Detection App")
st.markdown("Enter transaction details to predict the likelihood of fraud.")

# Loading the trained model and scaler
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the working directory.")
    st.stop()

# Defining the expected feature order based on creditcard.csv
feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Creating input form for transaction data
st.header("Transaction Input")
with st.form("transaction_form"):
    # Input for Time and Amount
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0, step=1.0)
    amount = st.number_input("Amount (transaction amount)", min_value=0.0, value=0.0, step=0.01)

    # Input for V1–V28 features
    v_features = {}
    for i in range(1, 29):
        v_features[f"V{i}"] = st.number_input(f"V{i} (PCA feature)", value=0.0, step=0.01)

    # Submit button
    submitted = st.form_submit_button("Predict Fraud Probability")

# Processing input and prediction
if submitted:
    # Creating input DataFrame with explicit column order
    input_data = pd.DataFrame({
        'Time': [time],
        'Amount': [amount],
        **{f'V{i}': [v_features[f'V{i}']] for i in range(1, 29)}
    })

    # Reordering columns to match training data
    input_data = input_data[feature_order]

    # Scaling the input data
    try:
        input_scaled = scaler.transform(input_data)
    except ValueError as e:
        st.error(f"Scaler error: {e}")
        st.stop()

    # Predicting fraud probability
    fraud_prob = model.predict_proba(input_scaled)[0][1] * 100

    # Displaying result
    st.header("Prediction Result")
    st.write(f"**Fraud Probability**: {fraud_prob:.2f}%")
    if fraud_prob > 30:
        st.warning("High likelihood of fraud! Review this transaction carefully.")
    else:
        st.success("Low likelihood of fraud.")

# Adding instructions
st.markdown("""
### Instructions
1. Enter the transaction details (Time, Amount, and V1–V28).
2. Click **Predict Fraud Probability** to see the result.
3. The model uses a trained Random Forest classifier to estimate the likelihood of fraud.
""")