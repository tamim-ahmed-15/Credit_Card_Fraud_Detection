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
---
## Methodology and Model Implementation

The project was structured with specific tasks and models implemented each day. Below is a detailed breakdown, including observations and performance comparisons.

## Day 1: Data Exploration
- **Tasks**:
  - Loaded `creditcard.csv` with Pandas.
  - Analyzed class imbalance (284,315 non-fraud, 492 fraud).
  - Visualized features (V14, V10, V3, V4) using Seaborn box and density plots.
- **Observations**:
  - Severe imbalance (0.17% fraud) requires sampling techniques.
  - Key features (V14, V10) showed distinct fraud patterns.

## Day 2: Baseline Models
- **Tasks**:
  - Preprocessed data (StandardScaler, train/test split).
  - Trained Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes.
- **Models Used**:
  - Logistic Regression (AUC ~0.75–0.80).
  - Decision Tree (AUC ~0.80–0.85).
  - KNN (AUC ~0.78–0.83).
  - Gaussian Naive Bayes (AUC ~0.70–0.75).
- **Observations**:
  - Low recall due to class imbalance.
  - Decision Tree and KNN performed better but had high false negatives.

## Day 3: Model Improvement
- **Tasks**:
  - Applied undersampling (~788 balanced rows).
  - Trained Random Forest, Decision Tree, Gradient Boosting.
  - Analyzed feature importance (V14, V10, V3, V4).
- **Models Used**:
  - Random Forest (AUC ~0.90–0.95).
  - Decision Tree (AUC ~0.85–0.90).
  - Gradient Boosting (AUC ~0.90–0.95, recall ~0.75–0.90).
- **Observations**:
  - Undersampling improved recall; Gradient Boosting excelled for fraud detection.
  - Random Forest was robust for deployment.

## Day 4: Streamlit Web App
- **Tasks**:
  - Built `app.py`, a Streamlit app for batch fraud prediction via CSV upload.
  - Used Random Forest model (`random_forest_model.pkl`) and scaler (`scaler.pkl`).
  - Added CSV upload for transactions (Time, Amount, V1–V28) with fraud count display.
  - Included feature importance visualization (e.g., V14, V10).
  - Limited display to 100 rows for large CSVs, with full results downloadable.
  - Deployed locally
- **Models Used**:
  - Random Forest (AUC ~0.90–0.95).
- **Observations**:
  - CSV upload with fraud count improved usability.
  - Fixed Pandas Styler error for large CSVs (~303,794 rows) by increasing cell limit and limiting display.
  - Resolved `ValueError` by aligning feature order (Time, V1–V28, Amount).
- **Key Insights**:
  - CSV upload with fraud count streamlined batch processing and improved usability.
  - Fixed Pandas Styler error for large CSVs (~303,794 rows) by increasing cell limit to 10,000,000 and limiting displayed rows to 100.
  - Resolved a `ValueError` by aligning input feature order with training data (Time, V1–V28, Amount).
  - Non-zero probabilities for all-zero inputs were debugged, attributed to scaler transformations.
  - Screenrecord saved in the path `Day04/Streamlit_app.mp4` of this repository.

### Model Performance Comparison

The following table compares the performance of all models implemented across Days 2 and 3, evaluated on the test set:

| Model                  | AUC       | Precision | Recall    | Observations                                      |
|------------------------|-----------|-----------|-----------|--------------------------------------------------|
| Logistic Regression    | ~0.75–0.80| ~0.65–0.75| ~0.50–0.65| Poor recall due to class imbalance sensitivity.   |
| Decision Tree (Day 2)  | ~0.80–0.85| ~0.70–0.80| ~0.60–0.75| Moderate performance; high false negatives.       |
| KNN                    | ~0.78–0.83| ~0.65–0.75| ~0.55–0.70| Scaling-dependent; computationally intensive.     |
| Gaussian Naive Bayes   | ~0.70–0.75| ~0.60–0.70| ~0.50–0.65| Poor fit due to feature independence assumption.  |
| Decision Tree (Day 3)  | ~0.85–0.90| ~0.75–0.85| ~0.65–0.80| Improved with undersampling but outperformed by ensembles. |
| Random Forest          | ~0.90–0.95| ~0.80–0.95| ~0.70–0.85| High precision and AUC; robust and balanced.      |
| Gradient Boosting      | ~0.90–0.95| ~0.75–0.90| ~0.75–0.90| High recall, best for minimizing false negatives. |

**Overall Key Insights**:
- **Day 2 Models**: All baseline models (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes) struggled with class imbalance, resulting in low recall and high false negatives.
- **Day 3 Models**: Undersampling improved performance, with Random Forest and Gradient Boosting achieving superior AUC (~0.90–0.95). Gradient Boosting excelled in recall, critical for fraud detection.
- **Random Forest** was selected for the Streamlit app due to its balanced performance and robustness.
- Class imbalance was the primary challenge, mitigated by undersampling but at the cost of reduced training data.
