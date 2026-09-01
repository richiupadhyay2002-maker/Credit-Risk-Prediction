# 🏦 Credit Risk Prediction System

### *Machine Learning Models for Assessing Loan Default Risk*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

> ⚠️ **Note on this README**: The performance numbers below were corrected on 2 September 2026 to match the actual output of `03_model_testing_evaluation.ipynb`. An earlier version of this README overstated the results — see [Corrected vs. Original](#-corrected-vs-original) at the bottom.

---

## 📋 Table of Contents

- [💡 Problem Statement](#-problem-statement)
- [📊 Dataset Overview](#-dataset-overview)
- [🧠 Models Implemented](#-models-implemented)
- [🏆 Performance Comparison](#-performance-comparison)
- [⚡ Hyperparameter Tuning](#-hyperparameter-tuning)
- [🔑 Key Insights](#-key-insights)
- [🚀 How to Run](#-how-to-run)
- [📁 Project Structure](#-project-structure)
- [✅ Corrected vs. Original](#-corrected-vs-original)
- [👤 Author](#-author)

---

## 💡 Problem Statement

> *"How can financial institutions accurately predict whether a loan applicant will default, minimizing financial losses while maximizing approval rates for creditworthy individuals?"*

This project addresses **credit risk assessment** by benchmarking **7 machine learning algorithms** to classify loan applicants as **low-risk (fully pay)** or **high-risk (default)** based on their financial and demographic attributes.

---

## 📊 Dataset Overview

The dataset contains **32,581 loan applications** with **12 features** capturing applicant demographics, financial history, and loan characteristics. Target variable: `loan_status` (0 = Paid, 1 = Defaulted).

---

## 🧠 Models Implemented

| # | Model | Category |
|---|-------|----------|
| 1 | Naive Bayes | Probabilistic |
| 2 | Decision Tree | Tree-based |
| 3 | **Random Forest** | Ensemble — best performer |
| 4 | K-Nearest Neighbors | Instance-based |
| 5 | Logistic Regression | Linear |
| 6 | SVM (RBF Kernel) | Margin-based |
| 7 | Neural Network (MLP) | Deep Learning |

---

## 🏆 Performance Comparison

*(actual output of `03_model_testing_evaluation.ipynb`, run [DATE])*

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 | **Random Forest** | **92.91%** | 94.36% | 71.78% | **81.53%** | **93.16%** |
| 🥈 | Decision Tree | 91.45% | 88.57% | 69.81% | 78.08% | 90.24% |
| 🥉 | SVM (RBF Kernel) | 89.44% | 78.61% | 70.87% | 74.54% | 90.55% |
| 4 | Neural Network (MLP) | 88.23% | 73.13% | 72.77% | 72.95% | 89.85% |
| 5 | Naive Bayes | 84.24% | 63.44% | 65.45% | 64.43% | 85.26% |
| 6 | Logistic Regression | 82.72% | 57.70% | 77.76% | 66.25% | 88.28% |
| 7 | K-Nearest Neighbors | 81.24% | 54.93% | 78.04% | 64.48% | 87.68% |

**Best model: Random Forest — 92.91% accuracy, 81.53% F1-score.**

---

## ⚡ Hyperparameter Tuning

### Random Forest — Grid Search Results

```
Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}
Best CV Accuracy: 94.72%
Tuned Test Accuracy: 92.85%
Tuned Test ROC-AUC: 93.21%
```

> Note: tuning improved cross-validated accuracy but did **not** meaningfully improve held-out test accuracy (92.85% vs. 92.91% untuned) — a useful, honest finding about the limits of hyperparameter search on this dataset.

### Feature Importance — Top Predictors

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `loan_percent_income` | 22.3% |
| 2 | `loan_int_rate` | 18.7% |
| 3 | `loan_grade` | 15.2% |
| 4 | `person_income` | 11.8% |
| 5 | `loan_amnt` | 8.4% |

---

## 🔑 Key Insights

- **Loan-to-Income Ratio** is the single strongest predictor of default.
- Tree-based ensemble methods (Random Forest) outperform linear and instance-based models on this dataset.
- Hyperparameter tuning raised cross-validated accuracy but had negligible effect on test accuracy — worth investigating further (e.g. more granular search, different resampling strategy) before claiming a tuning win.

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn yellowbrick
jupyter notebook 01_eda.ipynb
jupyter notebook 02_feature_engineering.ipynb
jupyter notebook 03_model_testing_evaluation.ipynb
```

---

## 📁 Project Structure

```
credit_risk/
├── 01_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_model_testing_evaluation.ipynb
├── credit_risk_dataset.csv          # 32,581 records
└── README.md
```

---

## ✅ Corrected vs. Original

| Metric | Original README | Actual Notebook Output |
|--------|-----------------|-------------------------|
| Random Forest accuracy | 93.70% | **92.91%** |
| Tuned accuracy | 94.30% | **92.85%** (test) / 94.72% (CV only) |

The original README's numbers didn't match the notebook's actual printed output. This version reflects what the code actually produced.

---

## 👤 Author

**Richi Upadhyay** — Data Science & Machine Learning
