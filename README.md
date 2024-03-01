# Telco Customer Churn Prediction

## Overview

This project predicts customer churn for a telecommunications company using various machine learning models.

## Data and Variables

- `CustomerId`: Unique customer identifier
- `Gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether the customer is a senior citizen (1 for Yes, 0 for No)
- `Partner`: Whether the customer has a partner (Yes/No)
- `Dependents`: Whether the customer has dependents (Yes/No)
- `Tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Whether the customer has phone service (Yes/No)
- `MultipleLines`: Whether the customer has multiple lines (Yes/No/No Phone Service)
- `InternetService`: Customer's internet service provider (DSL, Fiber optic, No)
- `OnlineSecurity`: Whether the customer has online security (Yes, No, No internet service)
- `OnlineBackup`: Whether the customer has online backup (Yes, No, No internet service)
- `DeviceProtection`: Whether the customer has device protection (Yes, No, No internet service)
- `TechSupport`: Whether the customer has tech support (Yes, No, No internet service)
- `StreamingTV`: Whether the customer has streaming TV (Yes, No, No internet service)
- `StreamingMovies`: Whether the customer has streaming movies (Yes, No, No internet service)
- `Contract`: The length of the customer's contract (Month-to-month, One year, Two years)
- `PaperlessBilling`: Whether the customer has paperless billing (Yes, No)
- `PaymentMethod`: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- `MonthlyCharges`: The amount charged to the customer monthly
- `TotalCharges`: The total amount charged to the customer
- `Churn`: Whether the customer churned (Yes or No)

## Feature Engineering

- Engineered new features based on customer behavior, services used, and tenure.

## Model Selection and Training

Utilized models such as Logistic Regression, K-Nearest Neighbors, Random Forest, Gaussian Naive Bayes, XGBoost, and Decision Tree.

## Model Evaluation

- Split the data into training and testing sets.
- Applied encoding and scaling to features.
- Trained and evaluated models using accuracy, F1 score, and ROC AUC.
- Tuned hyperparameters using Randomized and Grid Search.

## Final Model Comparison

Compared final models based on accuracy, F1 score, and ROC AUC using cross-validation.
