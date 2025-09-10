# Credit Card Fraud Detection Project

**An Algorithmic Anomaly Detection Approach for Identifying Fraudulent Credit Card Transactions**

This project develops a machine learning pipeline to detect fraudulent credit card transactions using a highly imbalanced dataset from Kaggle. It focuses on maximizing recall for the fraud class while maintaining acceptable precision, helping minimize financial losses due to undetected fraud.

---

## Project Overview

Credit card fraud detection is challenging because fraudulent transactions are rare compared to legitimate ones. This repository addresses the challenge by building and evaluating a classification model with strategies for handling class imbalance and optimizing detection thresholds.

---

## Motivation

In fraud detection, missing a fraudulent transaction (false negative) is more costly than incorrectly flagging a legitimate one (false positive). This project prioritizes recall for fraud detection while balancing precision to manage false alarms.

---

## Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Key Characteristics:**

* Transactions from European cardholders over two days in September 2013.
* Features `V1` to `V28` are PCA-transformed for confidentiality.
* `Time` and `Amount` are original features.
* Target variable `Class`: fraud (1) or non-fraud (0).
* Highly imbalanced: Fraud ≈ 0.172% of transactions.

---

## Methodology

**Data Loading & Exploration:**

* Loaded dataset and analyzed structure, summary statistics, and class distribution.

**Data Preprocessing:**

* `RobustScaler` applied to `Time` and `Amount`.
* Stratified split: 75% train, 12.5% validation, 12.5% test.

**Model Training:**

* Baseline: Logistic Regression with `class_weight='balanced'` to handle class imbalance.
* Neural Network model implemented to capture non-linear patterns.

**Evaluation & Threshold Optimization:**

* Metrics: Precision, Recall, F1-score for fraud class.
* Precision-Recall curves and threshold iteration used to select optimal classification threshold.

---

## Key Findings

### Default Logistic Regression Model

* F1-score: 0.8257, Precision: 88.24%, Recall: 77.59%.
* High precision but recall is suboptimal (\~23% of fraudulent transactions missed).

### Class-Weighted Logistic Regression Model

* F1-score slightly improved recall to 84.48% but precision dropped to 58.33%.
* Trade-off too severe, resulting in many false alarms.

### Neural Network Model

* Optimal F1-score threshold: Precision \~81%, Recall \~81%, F1-score: 0.8103.
* Slightly lower F1 than default Logistic Regression, but provides a more balanced trade-off between precision and recall.

### Final Model Selection

* Neural Network chosen as the final model due to balanced performance.

* On unseen test set: Precision 79%, Recall 75%, F1-score 0.77.

* Slight drop from validation set expected; demonstrates minor overfitting but overall strong and reliable performance for real-world deployment.

* Logistic Regression, while a strong baseline, did not achieve the same level of balanced performance as the Neural Network.

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection-Project.git
cd Credit-Card-Fraud-Detection-Project
```

### 2. Set up Virtual Environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Kaggle Dataset

* Place `kaggle.json` in `~/.kaggle/`.
* Scripts in `src/` can download and unzip dataset automatically, or manually place `creditcard.csv` in `data/`.

### 5. Run Source Code

```bash
# Train model
python src/train_model.py

# Evaluate model
python src/evaluate_model.py

# Predict on new transactions
python src/predict.py --input data/sample_transactions.csv
```

*Scripts include usage instructions in their headers.*

---

## License

MIT License – see [LICENSE](LICENSE) for details.
