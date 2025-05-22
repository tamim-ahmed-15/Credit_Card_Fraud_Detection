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

### Day 1: Data Exploration
- **Tasks**:
  - Loaded `creditcard.csv` using Pandas and analyzed its structure.
  - Computed class distribution and feature statistics (mean, std, min, max).
  - Visualized features (e.g., V14, V10, V3, V4) using Seaborn to identify fraud patterns.
- **Observations**:
  - Confirmed severe class imbalance (99.83% non-fraud vs. 0.17% fraud).
  - Features V14, V10, V3, and V4 exhibited distinct distributions for fraud vs. non-fraud transactions.
  - Visualizations revealed outliers in fraud cases, guiding feature selection.
- **Models**: None implemented.

### Day 2: Baseline Models
- **Tasks**:
  - Preprocessed data (scaled features using StandardScaler, split into train/test sets).
  - Trained four baseline models to establish a performance benchmark.
- **Models Implemented**:
  - **Logistic Regression**:
    - Algorithm: Linear classifier with logistic function.
    - Parameters: Default (e.g., solver=’lbfgs’, C=1.0).
    - Metrics: AUC, Precision, Recall.
  - **Decision Tree**:
    - Algorithm: CART (Classification and Regression Tree).
    - Parameters: Default (e.g., max_depth=None).
    - Metrics: AUC, Precision, Recall.
  - **K-Nearest Neighbors (KNN)**:
    - Algorithm: Distance-based classifier.
    - Parameters: Default (e.g., n_neighbors=5, metric=’euclidean’).
    - Metrics: AUC, Precision, Recall.
  - **Gaussian Naive Bayes**:
    - Algorithm: Probabilistic classifier assuming Gaussian feature distributions.
    - Parameters: Default.
    - Metrics: AUC, Precision, Recall.
- **Observations**:
  - **Logistic Regression**: AUC ~0.75–0.80, with low recall (~0.50–0.65) due to class imbalance skewing predictions toward non-fraud.
  - **Decision Tree**: AUC ~0.80–0.85, moderate precision (~0.70–0.80), but low recall (~0.60–0.75) due to overfitting on imbalanced data.
  - **KNN**: AUC ~0.78–0.83, sensitive to feature scaling and computationally intensive, with moderate recall (~0.55–0.70).
  - **Gaussian Naive Bayes**: Lowest performance (AUC ~0.70–0.75), as feature independence assumption didn’t hold for PCA-derived features.
  - Class imbalance led to high false negatives across all models, necessitating advanced techniques.

### Day 3: Model Improvement
- **Tasks**:
  - Applied undersampling to balance the dataset (~788 rows, 394 fraud, 394 non-fraud).
  - Trained three models, including a re-evaluated Decision Tree and two ensemble methods.
  - Analyzed feature importance to identify key fraud indicators.
- **Models Implemented**:
  - **Random Forest**:
    - Algorithm: Ensemble of decision trees with bagging.
    - Parameters: Default (e.g., n_estimators=100).
    - Metrics: AUC, Precision, Recall.
  - **Decision Tree**:
    - Algorithm: CART, re-evaluated with balanced data.
    - Parameters: Default or tuned (e.g., max_depth=10).
    - Metrics: AUC, Precision, Recall.
  - **Gradient Boosting**:
    - Algorithm: Gradient Boosted Decision Trees (e.g., scikit-learn’s GradientBoostingClassifier or XGBoost).
    - Parameters: Default (e.g., n_estimators=100, learning_rate=0.1).
    - Metrics: AUC, Precision, Recall.
- **Observations**:
  - **Random Forest**: Achieved AUC ~0.90–0.95, with high precision (~0.80–0.95) and moderate recall (~0.70–0.85). Robust due to ensemble approach.
  - **Decision Tree**: Improved with undersampling (AUC ~0.85–0.90, precision ~0.75–0.85, recall ~0.65–0.80), but still outperformed by ensembles.
  - **Gradient Boosting**: AUC ~0.90–0.95, with slightly lower precision (~0.75–0.90) but higher recall (~0.75–0.90), ideal for fraud detection.
  - Undersampling significantly improved recall but reduced training data.
  - Feature importance highlighted V14, V10, V3, and V4 as top predictors.

### Day 4: Streamlit Web App
- **Tasks**:
  - Developed `app.py`, a Streamlit app for batch fraud prediction via CSV upload.
  - Loaded the trained Random Forest model (`random_forest_model.pkl`) and scaler (`scaler.pkl`).
  - Implemented CSV upload for transactions (Time, Amount, V1–V28), displaying predictions and a count of fraud cases.
  - Added a feature importance visualization to highlight key predictors (e.g., V14, V10).
  - Limited CSV prediction display to 100 rows for large files, with full results downloadable.
  - Deployed locally (`streamlit run app.py`, accessible at `http://localhost:8503/`).
- **Models Used**:
  - **Random Forest**: Deployed from Day 3 for its robust performance (AUC ~0.90–0.95).
- **Key Insights**:
  - CSV upload with fraud count streamlined batch processing and improved usability.
  - Fixed Pandas Styler error for large CSVs (~303,794 rows) by increasing cell limit to 10,000,000 and limiting displayed rows to 100.
  - Resolved a `ValueError` by aligning input feature order with training data (Time, V1–V28, Amount).
  - Non-zero probabilities for all-zero inputs were debugged, attributed to scaler transformations.
  - Screenrecord captured in `Day04/Streamlit_app.mp4`.

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

**Key Insights**:
- **Day 2 Models**: All baseline models (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes) struggled with class imbalance, resulting in low recall and high false negatives.
- **Day 3 Models**: Undersampling improved performance, with Random Forest and Gradient Boosting achieving superior AUC (~0.90–0.95). Gradient Boosting excelled in recall, critical for fraud detection.
- **Random Forest** was selected for the Streamlit app due to its balanced performance and robustness.
- Class imbalance was the primary challenge, mitigated by undersampling but at the cost of reduced training data.
