# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

## Overview

This repository contains a machine learning project to detect fraudulent credit card transactions using the `creditcard.csv` dataset. Developed over five days, the project encompasses data exploration, model training, a Streamlit web application for real-time predictions, and a comprehensive summary report. The system addresses the challenge of class imbalance (0.17% fraud cases) and achieves high performance with ensemble models.

**Key Objectives**:
- Analyze transaction data to identify fraud patterns.
- Train and optimize machine learning models for fraud detection.
- Deploy an interactive web app for user-friendly predictions.

---

## Dataset

The dataset is sourced from [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and is not included in this repository due to its size. Users must download it separately to run the notebook.

- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1–V28 PCA-derived features)
- **Target**: Class (0 = non-fraud, 1 = fraud)
- **Class Distribution**: 284,315 non-fraud (99.83%), 492 fraud (0.17%)

**Insight**: Severe class imbalance necessitated techniques like undersampling to improve model performance.

---

## Methodology

1. **Day 1: Data Exploration**
   - Analyzed class distribution and feature correlations.
   - Visualized key features (e.g., V14, V10) using Seaborn.
   - Identified class imbalance as a primary challenge.

2. **Day 2: Baseline Models**
   - Trained a Decision Tree classifier (AUC ~0.80–0.85).
   - Observed high false negatives due to class imbalance.

3. **Day 3: Model Improvement**
   - Applied undersampling (~788 rows, 394 fraud, 394 non-fraud).
   - Trained Random Forest (AUC ~0.90–0.95, precision ~0.80–0.95) and Gradient Boosting (AUC ~0.90–0.95, recall ~0.75–0.90).
   - Identified V14, V10, V3, V4 as top fraud indicators via feature importance.

4. **Day 4: Streamlit Web App**
   - Developed `app.py`, enabling users to input transaction data and receive fraud probability predictions.
   - Deployed locally via `streamlit run app.py` (accessible at `http://localhost:8501`).
   - Screenshot: [Streamlit_app.pdf]([streamlit_app_screenshot.png](https://github.com/tamim-ahmed-15/Credit_Card_Fraud_Detection/blob/main/Day04/Streamlit_app.pdf)).


### Model Performance

| Model              | AUC       | Precision | Recall    |
|--------------------|-----------|-----------|-----------|
| Decision Tree      | ~0.80–0.85| ~0.70–0.80| ~0.60–0.75|
| Random Forest      | ~0.90–0.95| ~0.80–0.95| ~0.70–0.85|
| Gradient Boosting  | ~0.90–0.95| ~0.75–0.90| ~0.75–0.90|

---

## Key Findings

- **High Model Performance**: Random Forest and Gradient Boosting achieved AUCs of ~0.90–0.95, with Gradient Boosting excelling in recall (~0.75–0.90), critical for fraud detection.
- **Feature Importance**: V14, V10, V3, and V4 were the most predictive features, highlighting the effectiveness of PCA-derived features.
- **Class Imbalance**: Undersampling balanced the dataset but reduced training data, suggesting SMOTE for future improvements.
- **Streamlit App**: The app provides an intuitive interface but requires manual input of 30 features, which could be streamlined with CSV uploads.
- **Challenges**:
  - Handled class imbalance through undersampling.
  - Resolved feature order mismatches in `app.py` (Day 4 `ValueError`).
  - Debugged non-zero probabilities (~10–20%) for all-zero inputs, attributed to scaler transformations.

---

## Repository Structure

| File                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `Credit_Card_Fraud_Analysis_Days1-4.ipynb` | Jupyter Notebook for Days 1–4 (exploration, modeling, app code) |
| `app.py`                          | Streamlit app for fraud prediction              |
| `random_forest_model.pkl`         | Trained Random Forest model                     |
| `scaler.pkl`                      | StandardScaler for feature scaling              |
| `streamlit_app_screenshot.png`    | Screenshot of Streamlit app                     |
| `summary_report.pdf`              | Final 1–2 page summary report                   |
| `README.md`                       | Project overview and setup instructions         |

**Note**: `creditcard.csv` is not included. Download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
