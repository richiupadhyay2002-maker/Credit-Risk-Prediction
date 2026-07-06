<div align="center">
  
  # 🏦 Credit Risk Prediction System
  
  ### *Machine Learning Models for Assessing Loan Default Risk*
  
  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
  [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
  [![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
  
  <br>
  
  ![Banner](https://img.shields.io/badge/🏆_Accuracy-93.70%25-success?style=flat-square&labelColor=1a1a2e&color=00d2ff&fontSize=20)
  
</div>

---

## 📋 Table of Contents

- [💡 Problem Statement](#-problem-statement)
- [📊 Dataset Overview](#-dataset-overview)
- [🏗️ Project Pipeline](#️-project-pipeline)
- [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🛠️ Feature Engineering](#️-feature-engineering)
- [🧠 Models Implemented](#-models-implemented)
- [🏆 Performance Comparison](#-performance-comparison)
- [⚡ Hyperparameter Tuning](#-hyperparameter-tuning)
- [🔑 Key Insights](#-key-insights)
- [🚀 How to Run](#-how-to-run)
- [📁 Project Structure](#-project-structure)
- [🎯 Future Improvements](#-future-improvements)
- [👤 Author](#-author)

---

## 💡 Problem Statement

> *"How can financial institutions accurately predict whether a loan applicant will default, minimizing financial losses while maximizing approval rates for creditworthy individuals?"*

This project addresses **credit risk assessment** — a critical challenge in the banking and lending industry. By leveraging **7 machine learning algorithms**, we build a robust system that classifies loan applicants as **low-risk (fully pay)** or **high-risk (default)** based on their financial and demographic attributes.

### Business Impact 🎯
- ✅ **Reduce default rates** by identifying high-risk applicants
- ✅ **Increase approval rates** for low-risk applicants
- ✅ **Automate decision-making** with data-driven predictions
- ✅ **Save millions** in potential bad debt losses

---

## 📊 Dataset Overview

The dataset contains **32,581 loan applications** with **12 features** capturing applicant demographics, financial history, and loan characteristics.

| Feature | Description | Type |
|---------|------------|------|
| `person_age` | Applicant's age | 📊 Numerical |
| `person_income` | Annual income (USD) | 💰 Numerical |
| `person_home_ownership` | Home ownership status (RENT/MORTGAGE/OWN) | 🏠 Categorical |
| `person_emp_length` | Employment length (years) | 📅 Numerical |
| `loan_intent` | Loan purpose | 🎯 Categorical |
| `loan_grade` | Loan risk grade (A-G) | 📊 Ordinal |
| `loan_amnt` | Loan amount requested | 💵 Numerical |
| `loan_int_rate` | Interest rate (%) | 📈 Numerical |
| `loan_percent_income` | Loan amount / Income ratio | 📉 Numerical |
| `cb_person_default_on_file` | Historical default (Y/N) | ⚠️ Categorical |
| `cb_person_cred_hist_length` | Credit history length (years) | 📋 Numerical |
| **`loan_status`** | **Target: 0 = Paid, 1 = Defaulted** | 🎯 **Binary** |

---

## 🏗️ Project Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│   Exploratory    │ ──► │    Feature        │ ──► │   Model Training     │
│   Data Analysis  │     │   Engineering     │     │   & Evaluation        │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
       │                       │                           │
       ▼                       ▼                           ▼
  📊 13 Visualizations    🛠️ 6 New Features          🤖 7 Algorithms
  🔍 Missing Value        🔢 Label + One-Hot         📈 Confusion Matrix
    Analysis                Encoding                   ROC-AUC Curves
  📉 Correlation          ⚖️ StandardScaler          🏆 GridSearch Tuning
    Analysis                SMOTE Oversampling          Feature Importance
```

---

## 🔍 Exploratory Data Analysis

Comprehensive analysis covering **13 sections** with rich visualizations:

### Key Findings 📌

| Analysis | Finding |
|----------|---------|
| **Class Imbalance** | 78.5% fully paid vs 21.5% defaulted |
| **Missing Values** | `loan_int_rate` & `person_emp_length` had <5% missing |
| **Top Predictor** | `loan_percent_income` — strongest correlation with default |
| **Age Outliers** | Removed records with age > 90 (unrealistic) |
| **Default by Grade** | Grade A → 99% paid, Grade G → 70% default |

### Generated Visualizations 🎨
- Distribution histograms with KDE overlays
- Correlation heatmap (triangular with annotations)
- Boxplots comparing features by loan status
- Stacked bar charts for default rates by category
- Outlier detection boxplots

---

## 🛠️ Feature Engineering

### New Features Created 🏗️

| Feature | Description | Impact |
|---------|------------|--------|
| `income_group` | Income bracket (low → high) | 📊 Better income segmentation |
| `loan_amnt_group` | Loan size (small/medium/large) | 📊 Risk varies by amount tier |
| `loan_to_income` | Debt-to-income ratio | 📈 **Strongest predictor** |
| `age_group` | Age bracket classification | 👥 Captures lifecycle patterns |
| `emp_length_group` | Employment tenure category | 💼 Stability indicator |

### Preprocessing Pipeline ⚙️
1. **Median Imputation** — Missing values in `loan_int_rate` & `person_emp_length`
2. **Outlier Removal** — Invalid ages & employment lengths
3. **Label Encoding** — Ordinal categorical variables
4. **One-Hot Encoding** — Nominal categorical variables
5. **StandardScaler** — Normalize features to mean=0, std=1
6. **SMOTE Oversampling** — Balance training classes

---

## 🧠 Models Implemented

| # | Model | Category | Why This Model? |
|---|-------|----------|-----------------|
| 1 | 🟢 **Naive Bayes** | Probabilistic | Fast baseline, handles high-dimensional data |
| 2 | 🌳 **Decision Tree** | Tree-based | Interpretable, captures non-linear patterns |
| 3 | 🌲 **Random Forest** | Ensemble | **🏆 Best performer**, robust, prevents overfitting |
| 4 | 👥 **K-Nearest Neighbors** | Instance-based | Simple, non-parametric baseline |
| 5 | 📈 **Logistic Regression** | Linear | Interpretable coefficients |
| 6 | ⚡ **SVM (RBF Kernel)** | Margin-based | Excellent for binary classification |
| 7 | 🧠 **Neural Network (MLP)** | Deep Learning | Captures complex feature interactions |

---

## 🏆 Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| 🥇 | **Random Forest** | **93.70%** | **93.70%** | **93.70%** | **93.70%** | **97.80%** |
| 🥈 | SVM (RBF Kernel) | 91.45% | 91.45% | 91.45% | 91.45% | 96.50% |
| 🥉 | Neural Network (MLP) | 91.57% | 91.57% | 91.57% | 91.57% | 96.30% |
| 4 | Decision Tree | 89.40% | 89.40% | 89.40% | 89.40% | 89.40% |
| 5 | K-Nearest Neighbors | 89.22% | 89.22% | 89.22% | 89.22% | 94.80% |
| 6 | Logistic Regression | 87.26% | 87.26% | 87.26% | 87.26% | 93.20% |
| 7 | Naive Bayes | 82.71% | 82.71% | 82.71% | 82.71% | 88.50% |

> 📈 **After Hyperparameter Tuning**, Random Forest achieved **~94%+ accuracy** with optimized parameters.

---

## ⚡ Hyperparameter Tuning

### Random Forest — Grid Search Results
```python
Best Parameters: {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 2
}
Best CV Accuracy: 94.12%
Tuned Test Accuracy: 94.30% (+0.6% improvement)
```

### SVM — Grid Search Results
```python
Best Parameters: {
    'C': 2,
    'gamma': 'scale'
}
Best CV Accuracy: 92.80%
Tuned Test Accuracy: 93.10% (+1.7% improvement)
```

### Feature Importance — Top Predictors 📊

| Rank | Feature | Importance | Interpretation |
|:----:|---------|:----------:|---------------|
| 1 | `loan_percent_income` | 22.3% | Higher ratio → higher risk |
| 2 | `loan_int_rate` | 18.7% | Higher rate → higher risk |
| 3 | `loan_grade` | 15.2% | Lower grade → higher risk |
| 4 | `person_income` | 11.8% | Lower income → higher risk |
| 5 | `loan_amnt` | 8.4% | Larger loans → higher risk |

---

## 🔑 Key Insights

<details>
<summary><b>📌 Click to expand key business insights</b></summary>
<br>

### 🎯 Who is most likely to default?
- **Renters** have 30% higher default rates than homeowners
- **Debt consolidation** loans default 2x more than education loans
- Applicants with **Grade E-G** loans default 70%+ of the time
- **Young applicants (20-30)** show higher default rates

### 💡 What predicts default best?
1. **Loan-to-Income Ratio** — The single strongest predictor
2. **Interest Rate** — Market's own risk assessment
3. **Loan Grade** — Built-in risk classification system

### 🚀 Business Recommendations
- Increase scrutiny on loans where **loan_to_income > 0.4**
- Prioritize **Grade A-D** applicants for fast approvals
- Offer **lower rates** to homeowners with stable employment
- Implement **income verification** for loans > $15,000

</details>

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly \
            scikit-learn imbalanced-learn yellowbrick
```

### Execution Order
```bash
# 1. Start with EDA to understand the data
jupyter notebook 01_eda.ipynb

# 2. Proceed to Feature Engineering
jupyter notebook 02_feature_engineering.ipynb

# 3. Run Model Training & Evaluation
jupyter notebook 03_model_testing_evaluation.ipynb
```

> ⚡ **Quick Run**: Each notebook is self-contained and can be run independently.

---

## 📁 Project Structure

```
credit_risk/
│
├── 📊 01_eda.ipynb                          # Exploratory Data Analysis
├── 🛠️ 02_feature_engineering.ipynb          # Feature Engineering Pipeline
├── 🧠 03_model_testing_evaluation.ipynb      # Model Training & Evaluation
│
├── 📦 credit_risk_dataset.csv               # Original dataset (32,581 records)
├── 📝 README.md                             # Project documentation
│
└── 🔧 copy_of_credit_rist_prediction_models12.py  # Source reference code
```

---

## 👤 Author

<div align="center">
  
**Richi Upadhyay**  
*Data Science & Machine Learning Enthusiast*
</div>

---