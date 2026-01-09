# 🏦 Credit Risk Assessment Model

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project simulates a banking environment to predict loan applicant default probabilities. Using **Logistic Regression**, the model analyzes financial behaviors (Income, Debt, Credit Score) to classify applicants as **"High Risk"** or **"Low Risk."**

This fulfills the need for **Quantitative Risk Assessment** and **Model Validation** in financial analytics.

## 🛠️ Technical Architecture
* **Language:** Python
* **Algorithms:** Logistic Regression (for interpretability), Random Forest (for comparison).
* **Key Libraries:** `pandas` (Data Manipulation), `scikit-learn` (Modeling), `matplotlib` (Visualization).

## 📊 Model Performance & Validation
The model was validated using an 80/20 Train-Test split on a synthetic dataset of 2,000 applicants.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **88.25%** | Percentage of correct predictions on unseen data. |
| **Precision** | **91.0%** | Ratio of correctly identified "Safe" loans. |
| **Recall** | **60.0%** | Ability to catch actual defaults (Risk Detection). |

### 🔍 Key Risk Drivers (Feature Importance)
The model identified the following variables as the strongest predictors of default:
1.  **Current Debt** (Positive Correlation to Risk)
2.  **Annual Income** (Negative Correlation to Risk)
3.  **Credit Score** (Negative Correlation to Risk)

## 🚀 How to Run
1. Clone the repository:
  
2.Install dependencies:
pip install pandas scikit-learn matplotlib

3.Run the analysis script:
python risk_model.py
