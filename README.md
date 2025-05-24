# Credit Card Fraud Detection (ML Developer Bootcamp)

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)

## Overview

It contains the work completed as part of the [ML Developer Bootcamp Milestone 1](https://www.notion.so/ML-Developer-Bootcamp-Milestone-1-1f7e0fadd2c880758320e27970a80716), focusing on developing a machine learning system to detect fraudulent credit card transactions using the `creditcard.csv` dataset. The project spans data exploration, training multiple baseline and advanced models, and deploying a Streamlit web application for fraud prediction.

**Key Objectives**:
- Explore and visualize transaction data to uncover fraud patterns.
- Train and compare a range of machine learning models for fraud detection.
- Deploy an interactive web app for user-friendly predictions.
---

## Dataset

The dataset is sourced from [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and is not included in this repository due to its size. Users must download it separately to run the notebook.

- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1–V28 PCA-derived features)
- **Target**: Class (0 = non-fraud, 1 = fraud)
- **Class Distribution**: 284,315 non-fraud (99.83%), 492 fraud (0.17%)
## Tools
  Python, pandas, numpy,matplotlib, seaborn, scikit-learn, streamlit,joblib.
## Methodology and Model Implementation

## Day 1: Data Exploration
- Loaded dataset with Pandas.
- Visualized features (V14, V10) using Seaborn.

## Day 2: Baseline Models
- Scaled features with StandardScaler.
- Trained Logistic Regression (AUC ~0.75–0.80), Decision Tree (AUC ~0.80–0.85), KNN (AUC ~0.78–0.83), Gaussian Naive Bayes (AUC ~0.70–0.75).

## Day 3: Model Improvement
- Applied undersampling (~788 rows).
- Trained Random Forest (AUC ~0.90–0.95), Decision Tree (AUC ~0.85–0.90), Gradient Boosting (AUC ~0.90–0.95, recall ~0.75–0.90).
- Analyzed feature importance (V14, V10).

## Day 4: Streamlit Web App
  - Built `app.py`, a Streamlit app for batch fraud prediction via CSV upload.
  - Used Random Forest model (`random_forest_model.pkl`) and scaler (`scaler.pkl`).
  - Added CSV upload for transactions (Time, Amount, V1–V28) with fraud count display.
  - Limited display to 100 rows for large CSVs, with full results downloadable.
  - Deployed locally
  - Screenrecord saved in the path `Day04/Streamlit_app.mp4` of this repository.

**Overall Key Insights**:
- **Day 2 Models**: All baseline models (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes) struggled with class imbalance, resulting in low recall and high false negatives.
- **Day 3 Models**: Undersampling improved performance, with Random Forest and Gradient Boosting achieving superior AUC (~0.90–0.95). Gradient Boosting excelled in recall, critical for fraud detection.
- **Random Forest** was selected for the Streamlit app due to its balanced performance and robustness.
- Class imbalance was the primary challenge, mitigated by undersampling but at the cost of reduced training data.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tamim-ahmed-15/Credit_Card_Fraud_Detection.git
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib streamlit joblib
   ```

3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook Credit_Card_Fraud_Analysis.ipynb
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
