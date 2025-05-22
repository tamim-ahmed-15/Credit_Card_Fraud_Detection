# Credit Card Fraud Detection

## Project Summary
This project develops a machine learning system to detect fraudulent credit card transactions using the `creditcard.csv` dataset. The goal is to build and deploy a predictive model via a Streamlit web app, addressing the challenge of class imbalance (492 fraud vs. 284,315 non-fraud cases). Over five days, we explored data, trained baseline and advanced models, built an interactive app, and summarized findings.

## Dataset
- **Source**: `creditcard.csv` (not included in repository due to size; available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)).
- **Size**: 284,807 transactions.
- **Features**: 30 (Time, Amount, V1–V28 PCA-derived features).
- **Target**: Class (0 = non-fraud, 1 = fraud, 0.17% fraud cases).
- **Key Insight**: Severe class imbalance required techniques like undersampling.

## Method
- **Day 1 (Data Exploration)**: Analyzed class distribution, visualized features (e.g., V14, V10), and confirmed imbalance (99.83% non-fraud).
- **Day 2 (Baseline Models)**: Trained a Decision Tree (AUC ~0.80–0.85), identified high false negatives due to imbalance.
- **Day 3 (Model Improvement)**:
  - Applied undersampling (~788 rows, 394 fraud, 394 non-fraud).
  - Trained Random Forest (AUC ~0.90–0.95, precision ~0.80–0.95) and Gradient Boosting (AUC ~0.90–0.95, recall ~0.75–0.90).
  - Key features: V14, V10, V3, V4 (via feature importance).
- **Day 4 (Streamlit App)**: Built `app.py` to input transaction data and predict fraud probability using Random Forest. Deployed locally (`http://localhost:8501`).
- **Day 5 (Final Review)**: Polished notebook, created this README, and wrote a summary report.

## Key Findings
- **Model Performance**: Random Forest and Gradient Boosting outperformed Decision Tree, with AUCs ~0.90–0.95. Gradient Boosting excelled in recall (~0.75–0.90), crucial for fraud detection.
- **Class Imbalance**: Undersampling improved recall but reduced training data, suggesting SMOTE for future work.
- **Feature Importance**: V14, V10, V3, V4 were critical fraud indicators.
- **Streamlit App**: Provided an interactive interface but manual input of 30 features was cumbersome; CSV upload could enhance usability.
- **Challenges**: Handling class imbalance, ensuring feature order in `app.py`, and debugging non-zero probabilities for zero inputs (due to scaler transformation).

## Repository Structure
- `Credit_Card_Fraud_Analysis_Days1-4.ipynb`: Combined notebook for Days 1–4 (exploration, modeling, app code).
- `app.py`: Streamlit app for fraud prediction.
- `random_forest_model.pkl`: Trained Random Forest model.
- `scaler.pkl`: StandardScaler for feature scaling.
- `streamlit_app_screenshot.png`: Screenshot of Streamlit app.
- `summary_report.pdf`: Final 1–2 page report (Day 5).

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd credit-card-fraud-analysis
