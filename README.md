# 💳 CreditWise — Loan Approval Prediction System


> A Machine Learning project that predicts loan approval using multiple classification algorithms, with a clean interactive frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FJS-green?logo=html5)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-%F0%9F%9A%80%20Streamlit-red)](https://creditwise--loan--approval-bzyw22crygmzoxnk85hhx7.streamlit.app)

---

## 📌 Project Overview

CreditWise is a loan approval prediction system built as part of a B.Tech Computer Science (AI & ML) project. It uses real-world financial features to predict whether a loan application should be approved or rejected.

---

## 🎯 Features

- Predicts loan approval based on 18+ applicant features
- Compares 3 ML models: Logistic Regression, Naive Bayes, KNN
- Interactive frontend with confidence score, key factors, and model comparison chart
- Handles missing values, feature engineering, and data preprocessing

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `Applicant_Income` | Monthly income of the applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Credit_Score` | Credit score (300–850) |
| `DTI_Ratio` | Debt-to-income ratio |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan repayment period (months) |
| `Savings` | Total savings of applicant |
| `Collateral_Value` | Value of collateral offered |
| `Existing_Loans` | Number of active loans |
| `Employment_Status` | Salaried / Self-employed / Unemployed |
| `Education_Level` | Graduate / Not Graduate |
| `Marital_Status` | Married / Single |
| `Gender` | Male / Female |
| `Dependents` | Number of dependents |
| `Employer_Category` | Government / Private / MNC |
| `Loan_Purpose` | Home / Car / Business / Personal / Education |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Approved` | **Target variable** (Yes / No) |

---

## 🤖 Models & Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **87.5%** | 79.0% | 80.3% | 79.7% |
| Naive Bayes | 86.5% | 78.3% | 77.0% | 77.7% |
| KNN (k=5) | 75.5% | 62.0% | 50.8% | 55.9% |

> ✅ **Best Model: Logistic Regression** with 87.5% accuracy

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Notebook:** Jupyter Notebook

---

## 📁 Project Structure

```
creditwise-loan-approval/
│
├── credit_wise_loan.ipynb   # Main ML notebook (EDA + preprocessing + models)
├── index.html               # Interactive frontend (no backend needed)
├── loan_approval_data.csv   # Dataset (add your own)
└── README.md
```

---

## 🚀 How to Run

### Run the Notebook
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook credit_wise_loan.ipynb
```

### Run the Frontend
Simply open `index.html` in any browser — no server required!

---

## 📈 Preprocessing Steps

1. Handled missing values (median for numeric, mode for categorical)
2. Label encoding for binary features
3. One-hot encoding for multi-class categorical features
4. Feature engineering: `Credit_Score_sq`, `DTI_Ratio_sq`
5. StandardScaler for normalization
6. Train-test split: 80% / 20%

---

## 👤 Author

**Gyanendra**
B.Tech Computer Science (AI & ML)
Axis Institute of Technology and Management, Kanpur (AKTU — 2023–2027)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
