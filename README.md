# Credit Card Fraud Dataset Analysis

This project analyzes the Credit Card Fraud Dataset to explore transaction patterns, perform data profiling, and visualize fraud trends. It was developed as part of an internship Day 1 assignment to understand fraud detection in financial transactions using Python, Pandas, Matplotlib, and Seaborn.
Project Overview
The project involves:

Data Profiling: Summarizing dataset structure, content, and quality (e.g., shape, types, missing values, outliers).
Exploration: Inspecting features like Time, Amount, and Class (0 = Normal, 1 = Fraud).
Visualizations: Creating at least three visualizations:
Histograms for Amount and Time.
Correlation matrix for feature relationships.
Count plot (log scale) and boxplot for fraud vs. normal transactions.


Fraud Trend Observations: Analyzing patterns like class imbalance (~0.17% fraud) and transaction amount differences.

The analysis informs machine learning strategies for fraud detection, supporting secure transactions and business value (e.g., reducing fraud losses).
Dataset

Source: Kaggle Credit Card Fraud Dataset
Size: ~284,807 transactions, 31 features (Time, V1–V28, Amount, Class).
Features:
Time: Seconds elapsed since first transaction.
V1–V28: Anonymized PCA-transformed features.
Amount: Transaction amount.
Class: 0 (normal), 1 (fraud).


Note: Due to size, creditcard.csv is not included in this repository. Download it from Kaggle and place it in the project directory.

Requirements

Python 3.8+
Libraries:pip install pandas matplotlib seaborn numpy


Jupyter Notebook:pip install notebook


Optional (for automated profiling):pip install ydata-profiling



Setup

Clone the Repository:
git clone https://github.com/your-username/credit-card-fraud-analysis.git
cd credit-card-fraud-analysis


Download the Dataset:

Get creditcard.csv from Kaggle.
Place it in the project directory.


Install Dependencies:
pip install -r requirements.txt  # If you create a requirements.txt

Or install manually:
pip install pandas matplotlib seaborn numpy notebook


Start Jupyter Notebook:
jupyter notebook



Usage

Open Credit_Card_Fraud_Analysis_With_Profiling.ipynb in Jupyter Notebook.
Ensure creditcard.csv is in the project directory (or update the path in the notebook).
Run all cells to:
Profile the dataset (structure, stats, outliers, etc.).
Generate visualizations (saved as PNGs).
View fraud trend observations.


Check the profile_report.html (if using ydata-profiling) for an interactive report.

Outputs

Data Profiling:
Shape: 284,807 rows, 31 columns.
No missing values, ~1,135 duplicates.
Outliers in Amount: ~25,000 (based on IQR).
Class imbalance: ~492 fraud (0.17%), ~284,315 normal.


Visualizations (saved as PNGs):
histograms.png: Amount (right-skewed), Time (bimodal).
correlation_matrix.png: Near-zero correlations (PCA features).
count_plot_log.png: Fraud vs. normal counts (log scale, annotated).
boxplot_amount.png: Amount by class (fraud has lower median).


Observations:
Severe class imbalance requires ML techniques like anomaly detection.
Fraud transactions favor smaller amounts, making Amount predictive.
Temporal patterns in Time suggest further analysis for fraud clustering.



Repository Structure
credit-card-fraud-analysis/
├── Credit_Card_Fraud_Analysis_With_Profiling.ipynb  # Jupyter Notebook
├── histograms.png                                  # Amount and Time histograms
├── correlation_matrix.png                          # Correlation heatmap
├── count_plot_log.png                              # Fraud vs. normal count plot
├── boxplot_amount.png                              # Amount by class boxplot
├── README.md                                      # This file
└── requirements.txt                               # Optional: dependencies

License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
For questions, contact [Your Name] at [your.email@example.com] or open an issue on GitHub.

Built for internship Day 1 to explore fraud detection and support ML-driven financial security.
