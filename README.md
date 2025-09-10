# **Credit Card Fraud Detection Project**

## **An Algorithmic Anomaly Detection Approach**

This repository contains a machine learning project focused on detecting fraudulent credit card transactions. Utilizing a highly imbalanced dataset from Kaggle, the project aims to develop a robust classification model that prioritizes the detection of actual fraudulent activities while managing false alarms.

### **Project Overview**

Credit card fraud poses significant financial risks to banks, merchants, and consumers. The challenge lies in accurately identifying these rare fraudulent transactions amidst a vast majority of legitimate ones. This project addresses this critical problem by building and evaluating a classification model, with a particular focus oon handling the extreme class imbalance inherent in fraud datasets.

### **Motivation**

The primary motivation behind this project is to minimize financial losses due to undetected fraud. In this domain, a **false negative** (missing an actual fraudulent transaction) is significantly more costly than a **false positive** (incorrectly flagging a legitimate transaction as fraud). Therefore, the model's performance is heavily evaluated on its ability to maximize **recall** for the fraud class, while maintaining an acceptable **precision**.

### **Dataset**

The dataset used in this project is the "Credit Card Fraud Detection" dataset available on Kaggle.

* **Source:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
* **Characteristics:**  
  * Contains transactions made by European cardholders over two days in September 2013\.  
  * Features V1 to V28 are the result of a PCA transformation due to confidentiality issues.  
  * Features Time and Amount are the only original features not transformed by PCA.  
  * The Class variable is the target, indicating fraud (1) or non-fraud (0).  
  * **Highly Imbalanced:** Fraudulent transactions constitute only a tiny fraction (approx. 0.172%) of all transactions.

### **Methodology (Current Progress)**

This notebook covers the initial phases of the machine learning pipeline:

1. **Data Loading & Initial Exploration:**  
   * The dataset is loaded, and its basic structure and summary statistics are examined.  
   * A critical analysis of the Class distribution is performed, confirming the severe class imbalance.  
2. **Data Preprocessing:**  
   * **Feature Scaling:** RobustScaler is applied to the Time and Amount features to handle their varying scales and the presence of outliers in the Amount column. This is performed after data splitting to prevent data leakage.  
   * **Data Splitting:** The dataset is rigorously split into 75% for training, 12.5% for validation, and 12.5% for testing. Crucially, stratify=y is used in train\_test\_split to ensure that the class distribution (especially the rare fraud class) is maintained proportionally across all three subsets.  
3. **Model Training: Logistic Regression (Baseline Model):**  
   * Logistic Regression is chosen as the initial baseline model due to its simplicity and interpretability.  
   * To directly address the class imbalance and prioritize recall for the fraud class, the class\_weight='balanced' parameter is utilized during model training. This automatically assigns higher weights to the minority class.  
4. **Model Evaluation & Threshold Optimization:**  
   * The model's performance is evaluated on the validation set, focusing on **Precision**, **Recall**, and **F1-score** for the 'Fraud' class, as overall accuracy is misleading for imbalanced data.  
   * A **Precision-Recall Curve** is generated to visualize the inherent trade-off between these two metrics across different classification probability thresholds.  
   * The project then iterates through a range of thresholds (0.00 to 1.00) to identify the specific precision and recall values achievable at each point. This allows for a data-driven decision on the optimal threshold based on the desired balance between minimizing false negatives and managing false positives.

### **Key Findings (Current)**

* The initial Logistic Regression model (without class\_weight='balanced') showed high accuracy but poor recall for the fraud class, as expected due to imbalance.  
* Applying class\_weight='balanced' significantly improved the recall for the 'Fraud' class (e.g., from \~0.76 to \~0.91), successfully reducing false negatives.  
* This improvement in recall came at the cost of a substantial drop in precision for the 'Fraud' class, highlighting the critical precision-recall trade-off.  
* The threshold iteration analysis provides a clear spectrum of precision-recall pairs, enabling a strategic choice of the classification threshold to align with the project's objective of minimizing false negatives.

### **How to Run**

1. **Clone the Repository:**  
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection-Project.git  
   cd Credit-Card-Fraud-Detection-Project

2. **Set up Virtual Environment (Recommended):**  
   python \-m venv .venv  
   \# On Windows:  
   .\\.venv\\Scripts\\activate  
   \# On macOS/Linux:  
   source .venv/bin/activate

3. **Install Dependencies:**  
   pip install \-r requirements.txt

   *(Note: You'll need to generate this file by running pip freeze \> requirements.txt in your activated environment after installing all libraries.)*  
4. **Download Kaggle Dataset:**  
   * Ensure you have a kaggle.json file (containing your Kaggle API credentials) in your \~/.kaggle/ directory.  
   * The notebook contains commands to download and unzip the creditcardfraud.zip dataset directly into the project folder. Alternatively, manually download creditcard.csv from the Kaggle dataset page and place it in a data/ subdirectory within your project.  
5. **Run the Jupyter Notebook:**  
   jupyter notebook Credit-Card-Fraud-Detection.ipynb

   Alternatively, open Credit-Card-Fraud-Detection.ipynb directly in VS Code with the Jupyter extension installed, or upload it to Google Colab.

### **Future Work**

* **Exploring More Complex Models:** Implement and compare the performance of other classification algorithms such as Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM), or Neural Networks.  
* **Advanced Resampling Techniques:** Experiment with techniques like SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN to further balance the training dataset.  
* **Feature Engineering:** Investigate the creation of new features from existing ones (e.g., transaction velocity, time-based aggregations) that might provide more predictive power.  
* **Unsupervised Anomaly Detection:** Explore unsupervised methods like Isolation Forest or One-Class SVM to detect novel fraud patterns without relying on labeled data.  
* **Model Deployment:** Consider the steps required to deploy the final model for real-time inference.

### **License**

This project is open-sourced under the MIT License. See the LICENSE file for more details.
