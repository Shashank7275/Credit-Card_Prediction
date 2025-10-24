# 💳 Credit Card Default Prediction

A Machine Learning web app built with **Logistic Regression** and **Streamlit** to predict whether a credit card customer will default on payment.

---

## 🚀 Project Overview
This project predicts credit card default using a **Logistic Regression** model.  
It includes data preprocessing, model training, evaluation, and a **Streamlit** web interface for real-time predictions.

---

## 🧠 Model Used
- **Algorithm:** Logistic Regression  
- **Framework:** scikit-learn  
- **Frontend:** Streamlit  
- **Language:** Python 3.9+  

---

## 📊 Dataset
You can use any credit card default dataset such as:
> UCI “Default of Credit Card Clients Dataset”

**Target column:** `default_payment_next_month` (0 = No Default, 1 = Default)

**Example features:**
- LIMIT_BAL — Credit Limit  
- AGE — Age of the Customer  
- EDUCATION — Education Level  
- MARRIAGE — Marital Status  
- BILL_AMT1–6 — Bill Amounts  
- PAY_AMT1–6 — Payment Amounts  

---

## ⚙️ Data Preprocessing
1. Load dataset  
2. Handle missing values  
3. Encode categorical features  
4. Scale numerical columns (StandardScaler)  
5. Split data into train/test sets  

---

## 🧩 Model Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
