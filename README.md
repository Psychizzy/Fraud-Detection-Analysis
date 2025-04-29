#  Credit Card Fraud Detection – Machine Learning Project

This project focuses on detecting fraudulent credit card transactions using machine learning. With fraud cases being extremely rare but highly impactful, the goal was to build a model that accurately identifies fraud while minimizing disruption to legitimate users.

---

##  Objective

To develop a reliable, scalable, and accurate machine learning model that detects fraudulent transactions with a strong balance between **precision** and **recall**.

---

## 📂 Dataset Overview

- **Records**: 284,807 transactions
- **Fraud cases**: 492 (~0.17%)
- **Features**:
  - `Time`, `Amount`: raw values
  - `V1` to `V28`: PCA-transformed components
  - `Class`: Target variable (0 = Non-Fraud, 1 = Fraud)

---

##  Key Challenges

-  **Severe class imbalance** (fraud = 0.17%)
-  **High recall needed** (missing fraud is costly)
-  **Low false positives required** (avoid flagging good users)
-  **Accuracy is misleading — we focus on precision, recall, and F1-score

---

##  Exploratory Data Analysis (EDA)

-  Fraud peaks around 2AM and 11AM
-  No fixed pattern in amount — fraud occurs at both high and low amounts
-  Strong correlations found between fraud and features `V10`, `V12`, `V14`
-  Extreme class imbalance visualized and addressed later

---

##  Preprocessing

- Standardized `Amount` and `Time` using `StandardScaler`
- Created `scaled_df` for consistent modeling input
- Used **Stratified Train-Test Split** to maintain class ratio in test set

---

##  Handling Imbalance

- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to the training set only
- Compared models trained with and without SMOTE
- Evaluated all results on the original test set (to simulate real-world conditions)

---

## Models Compared

| Model                       | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|-----------------------------|-------------------|----------------|------------------|
| Logistic Regression (Raw)   | 82.9%             | 64.0%          | 72.4%            |
| Logistic Regression + SMOTE | 9.5%              | 91.0%          | 17.3%            |
| **Random Forest (Raw)**     | **96.3%**         | **80.6%**      | **87.8%**        |
| Random Forest + SMOTE       | 85.1%             | 81.6%          | 83.3%            |

✅ **Final model chosen**: Random Forest trained on original (imbalanced) data

---

## 📈 Final Model: Random Forest Classifier


RandomForestClassifier(n_estimators=100, random_state=42)

- Trained on original data (no oversampling)
- Achieved best precision/recall tradeoff
- Only 3 false positives and caught 80% of frauds
- Scalable, interpretable, and production-friendly

## Business Impact

- Detects majority of fraudulent transactions early

- Protects customers with minimal disruption

- Supports security teams with highly accurate alerts

- Ready for real-time deployment or batch scoring


## Tools Used
- Python
- Jupyter Notebook
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- SMOTE (Imbalanced-learn)

## Project Structure
fraud-detection/
├── notebooks/
│   └── fraud_detection_notebook.ipynb
├── data/
│   └── creditcard.csv
├── visuals/
│   └── (EDA plots, confusion matrices, etc.)
├── models/
│   └── rf_final_model.pkl
├── presentation/
│   └── Fraud_Detection_Presentation_Ajayi_Oluwaseyi.pptx
├── README.md
└── requirements.txt
