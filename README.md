# Credit Risk Prediction Using Machine Learning

## Project Overview

This project builds a machine learning model to predict **credit risk (Good vs Bad customers)** using banking customer data.

The goal is to help financial institutions:

• Identify high-risk customers likely to default
• Improve credit approval decisions
• Support credit limit assignment
• Improve portfolio risk management

The model is evaluated using **AUC and Gini coefficient**, which are standard metrics in credit risk modeling.

---

# Data Sources

The data was extracted from a banking database consisting of three main tables:

### 1. Customer Accounts Table

Contains account-level information such as:

* account open date
* last payment date
* credit limit
* balance amount
* payment history

Each customer may have **multiple accounts**.

### 2. Customer Enquiries Table

Contains credit enquiry records including:

* enquiry date
* enquiry amount
* enquiry purpose

Each customer may have **multiple enquiries**.

### 3. Customer Demographics Table

Contains customer-level information and the target variable.

Important columns include:

* customer demographic features
* engineered features
* **Bad_label** (target variable)

Each customer has **one record in this table**.

---

# Data Engineering Challenges

## Multi-table Data Structure

Account and enquiry tables contain multiple rows per customer, while the target label exists at the customer level.

### Solution

Customer-level aggregation was performed using:

```
groupby(customer_no)
```

Features created included:

* number of accounts
* total credit limit
* total balance
* enquiry counts
* repayment behaviour features

---

## Invalid Date Values

Some date columns contained invalid values such as:

```
0000-00-00
```

### Solution

Dates were parsed using:

```
pd.to_datetime(errors="coerce")
```

Invalid values were converted to missing timestamps.

---

## Non-Numeric Amount Fields

Some enquiry amount fields contained non-numeric characters.

### Solution

Regex cleaning was applied before converting to numeric values.

---

## Payment History Feature Engineering

Payment history fields contained encoded repayment patterns such as:

```
STD STD STD XXX 030 060 090
```

Where:

STD / 000 → No delay
030 → 30 days past due
060 → 60 days past due
090 → 90 days past due

These strings were parsed into numeric delinquency indicators.

New features created:

* worst delinquency
* average delinquency
* delinquency counts
* months since last delinquency
* recent delinquency indicators

These features significantly improved model performance.

---

# Handling Missing Values

After merging the aggregated datasets, missing values appeared for customers without accounts or enquiries.

These values were filled using:

```
df.fillna(0)
```

This approach preserves customers while treating absence of activity as meaningful information.

---

# Handling Class Imbalance

The dataset contained a strong imbalance:

Bad customers ≈ 4–5%
Good customers ≈ 95%

To address this:

* class weights were used in Logistic Regression
* scale_pos_weight was applied in boosting models

This ensures the model learns minority class behaviour.

---

# Machine Learning Models

Three main models were evaluated.

### Logistic Regression

Baseline model used for interpretability.

### XGBoost

Tree-based gradient boosting model capable of capturing complex nonlinear relationships.

### LightGBM

Efficient gradient boosting framework optimized for large datasets.

---

# Hyperparameter Tuning

Hyperparameters were optimized using **Optuna**.

Parameters tuned include:

* learning rate
* tree depth
* subsampling
* regularization
* feature fraction

This significantly improved model performance.

---

# Model Evaluation

Primary evaluation metrics:

• AUC (Area Under ROC Curve)
• Gini coefficient

Relationship:

```
Gini = 2 × AUC − 1
```

Results:

| Model               | AUC      | Gini        |
| ------------------- | -------- | ----------- |
| Logistic Regression | Moderate | Moderate    |
| LightGBM            | Good     | Competitive |
| XGBoost             | ~0.70    | ~0.41       |

A **Gini above 0.40** indicates strong discriminatory power in credit scoring.

---

# Decile Analysis

Customers were ranked by predicted risk score.

Observations:

* Highest risk decile captured the largest concentration of bad customers
* Bad rate decreased steadily across deciles
* Clear rank ordering confirmed model reliability

---

# Key Predictive Features

Top features identified by model importance:

* worst_dpd_12m
* dpd_30_plus_count
* utilisation_ratio
* total_credit_limit
* months_since_last_payment
* enquiry_count_90d
* number_of_accounts
* average_payment_gap

These features represent customer repayment behaviour and credit usage patterns.

---

# Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
Optuna
Matplotlib
Seaborn
MySQL

---

# Key Takeaways

• Feature engineering from payment history significantly improved model performance
• Aggregating multi-table banking data to customer level is essential
• Gradient boosting models outperform linear models for credit risk prediction
• Hyperparameter optimization improves model stability and accuracy

---

# Author

Yussouf R
Data Science / Machine Learning Project
