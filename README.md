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
- ... (other variables)

## Feature Engineering

- Engineered new features based on customer behavior, services used, and tenure.

## Model Selection and Training

Utilized models such as Logistic Regression, K-Nearest Neighbors, Random Forest, Gaussian Naive Bayes, XGBoost, and Decision Tree.

## Model Evaluation

- Split the data into training and testing sets.
- Applied encoding and scaling to features.
- Trained and evaluated models using accuracy, F1 score, and ROC AUC.
- Tuned hyperparameters using Randomized Search (KNN and Random Forest) and Grid Search (Logistic Regression and XGBoost).

## Final Model Comparison

Compared final models based on accuracy, F1 score, and ROC AUC using cross-validation.
