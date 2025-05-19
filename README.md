# Credit Card Fraud Dataset Analysis

![Fraud vs. Normal Count Plot](count_plot_log.png)

This project analyzes the [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to explore transaction patterns, perform data profiling, and visualize fraud trends. It was developed as part of an internship Day 1 assignment to understand fraud detection in financial transactions using Python, Pandas, Matplotlib, and Seaborn.

## Project Overview

The project involves:
- **Data Profiling**: Summarizing dataset structure, content, and quality (e.g., shape, types, missing values, outliers).
- **Exploration**: Inspecting features like `Time`, `Amount`, and `Class` (0 = Normal, 1 = Fraud).
- **Visualizations**: Creating at least three visualizations:
  - Histograms for `Amount` and `Time`.
  - Correlation matrix for feature relationships.
  - Count plot (log scale) and boxplot for fraud vs. normal transactions.
- **Fraud Trend Observations**: Analyzing patterns like class imbalance (~0.17% fraud) and transaction amount differences.

The analysis informs machine learning strategies for fraud detection, supporting secure transactions and business value (e.g., reducing fraud losses).

## Dataset

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~284,807 transactions, 31 features (Time, V1–V28, Amount, Class).
- **Features**:
  - `Time`: Seconds elapsed since first transaction.
  - `V1–V28`: Anonymized PCA-transformed features.
  - `Amount`: Transaction amount.
  - `Class`: 0 (normal), 1 (fraud).
- **Note**: Due to size, `creditcard.csv` is not included in this repository. Download it from Kaggle and place it in the project directory.

## Outputs

### Data Profiling
- **Shape:** 284,807 rows, 31 columns  
- **Missing Values:** None  
- **Duplicates:** Approximately 1,135 duplicate records  
- **Outliers in Amount:** Around 25,000 transactions identified as outliers based on Interquartile Range (IQR)  
- **Class Imbalance:**  
  - Fraudulent transactions: ~492 (0.17%)  
  - Normal transactions: ~284,315 (99.83%)  

### Visualizations (saved as PNG files)

| Filename             | Description                                      |
| -------------------- | ------------------------------------------------|
| `histograms.png`     | Amount distribution (right-skewed), Time (bimodal) |
| `correlation_matrix.png` | Features mostly show near-zero correlations (PCA features) |
| `count_plot_log.png` | Fraud vs. normal counts (log scale, annotated)  |
| `boxplot_amount.png` | Amount by class (fraud has lower median)         |

### Observations
- The dataset exhibits **severe class imbalance**, requiring specialized machine learning techniques such as anomaly detection or resampling to effectively detect fraud.  
- Fraudulent transactions typically involve **smaller amounts**, making the Amount feature a useful predictor.  
- Temporal patterns in the Time feature suggest potential for further analysis to detect clusters or bursts of fraudulent activity.


## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install pandas matplotlib seaborn numpy
