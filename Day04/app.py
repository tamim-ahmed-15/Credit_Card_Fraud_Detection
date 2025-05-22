import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Increasing Pandas Styler cell limit
pd.set_option("styler.render.max_elements", 10000000)

# Set page title
st.title("Credit Card Fraud Detection")

# Loading model and scaler
try:
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# Defining expected feature order
FEATURES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# --- CSV Upload Section ---
st.header("Fraud Prediction via CSV Upload")
st.write("Upload a CSV file with columns: Time, V1, V2, ..., V28, Amount")
st.info("For large files, only the first 100 rows are displayed below. Download the full results using the button provided.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        if not all(col in df.columns for col in FEATURES):
            st.error(f"CSV must contain columns: {', '.join(FEATURES)}")
        else:
            # Check row count for performance warning
            if len(df) > 500000:
                st.warning(f"CSV contains {len(df)} rows. Processing large files may be slow. Consider splitting the file.")
            
            # Reorder columns to match FEATURES
            df = df[FEATURES]
            
            # Scale data
            scaled_data = scaler.transform(df)
            
            # Predict probabilities
            probs = model.predict_proba(scaled_data)[:, 1]
            predictions = ["Fraud" if p >= 0.95 else "Non-Fraud" for p in probs]
            
            # Count fraud cases
            fraud_count = sum(1 for p in predictions if p == "Fraud")
            total_count = len(predictions)
            
            # Create result dataframe
            result_df = df.copy()
            result_df["Fraud_Probability"] = probs
            result_df["Prediction"] = predictions
            
            # Display fraud count
            st.subheader("Prediction Summary")
            st.metric(label="Fraud Cases Detected", value=f"{fraud_count} / {total_count}")
            
            # Display limited results
            display_limit = 100
            st.subheader("Prediction Results")
            st.write(f"Showing first {min(display_limit, len(result_df))} rows of {len(result_df)} total rows.")
            st.dataframe(result_df.head(display_limit).style.format({"Fraud_Probability": "{:.2%}"}))
            
            # Option to download full results
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Predictions",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")