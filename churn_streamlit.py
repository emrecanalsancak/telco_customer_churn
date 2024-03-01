import streamlit as st

st.set_page_config(layout="wide")
st.markdown(
    """<h1 style='color: #3498db; text-align: center;'>TELCO Customer Churn Prediction</h1>""",
    unsafe_allow_html=True,
)


tab_home, model_tab, conc_tab = st.tabs(
    [
        "Home",
        "Model Training",
        "Conclusion",
    ]
)


tab_home.subheader("What is the project about?")
tab_home.write(
    "This project predicts customer churn for a telecommunications company using various machine learning models."
)
tab_home.write(
    "Telco customer churn data contains information about a fictitious telecommunications company that provides home phone and internet services to 7,043 customers in California in the third quarter. It shows which customers have left the service, stayed, or signed up for the service."
)


tab_home.subheader("Objective📝")

objectives = [
    "Data Exploration and Preprocessing",
    "Feature Engineering",
    "Model Building",
    "Model Evaluation",
]

for objective in objectives:
    tab_home.markdown(f"- {objective}")

tab_home.subheader("Data and Variables")

tab_home.warning(
    "Since this is the company's private dataset, I cannot share the complete dataset. Only a subset of the data variables and their general meanings are provided below."
)

tab_home.write(
    """
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
- `Churn`: Whether the customer churned (Yes or No)"""
)

# Model Tab
model_tab.subheader("Model Development Overview")
model_tab.write(
    "<span style='color: orangered;'>This section displays the different model's training process.</span>",
    unsafe_allow_html=True,
)

with model_tab.expander("Base Model Comparison"):
    st.image("plots/model_comparison.png")


model_tab.subheader("Logistic Regression")
with model_tab.expander("Roc Curve"):
    st.image("plots/log_res_roc_curve.png")

with model_tab.expander("Confusion Matrix"):
    st.image("plots/log_res_conf.png")

with model_tab.expander("Feature Importance"):
    st.image("plots/log_res_feature_imp.png")

model_tab.subheader("K-Nearest Neighbors")
with model_tab.expander("Roc Curve"):
    st.image("plots/knn_roc_curve.png")

with model_tab.expander("Confusion Matrix"):
    st.image("plots/knn_roc_curve.png")

with model_tab.expander("Hyperparameter Tuning"):
    st.image("plots/knn_hyperparam.png")

model_tab.subheader("Random Forest Classifier")
with model_tab.expander("Roc Curve"):
    st.image("plots/rf_roc_curve.png")

with model_tab.expander("Confusion Matrix"):
    st.image("plots/random_forest_conf.png")

with model_tab.expander("Feature Importance"):
    st.image("plots/rf_feature_imp.png")

model_tab.subheader("XGBoost")
with model_tab.expander("Roc Curve"):
    st.image("plots/xgb_roc_curve.png")

with model_tab.expander("Confusion Matrix"):
    st.image("plots/xgb_conf.png")

with model_tab.expander("Feature Importance"):
    st.image("plots/xgb_feature_imp.png")

# Conclusion Tab

conc_tab.subheader("Model Performance")

conc_tab.write(
    """
Among the models evaluated, Logistic Regression and XGBoost performed the best in terms of accuracy, F1-score, and area under the ROC curve (AUC-ROC).
Logistic Regression achieved an accuracy of approximately 81% on the test data, with an F1-score of around 0.62 and an AUC-ROC of 0.85.
XGBoost also achieved an accuracy of approximately 81% on the test data, with an F1-score of around 0.57 and an AUC-ROC of 0.84.
"""
)

conc_tab.subheader("Model Comparison")

conc_tab.write(
    """
Logistic Regression and XGBoost outperformed other models such as KNN, RandomForest, GaussianNB, and DecisionTreeClassifier.
While RandomForest and XGBoost had comparable performance to Logistic Regression, they exhibited slightly lower accuracy and F1-score.
"""
)

with conc_tab.expander("Final Model Comparison", expanded=True):
    st.image("plots/final_model_comparison.png")

conc_tab.subheader("Hyperparameter Tuning")

conc_tab.write(
    """
Hyperparameter tuning was performed for models such as KNN, Logistic Regression, RandomForest, and XGBoost to optimize their performance.
RandomizedSearchCV and GridSearchCV were utilized to find the best hyperparameters for each model.
"""
)

conc_tab.subheader("Visualizations")

conc_tab.write(
    """
ROC curves were plotted to visualize the performance of models in terms of true positive rate (sensitivity) versus false positive rate (1-specificity).
Confusion matrices were plotted to visualize the number of true positives, true negatives, false positives, and false negatives for each model.
"""
)

conc_tab.subheader("Final Recommendations")

conc_tab.write(
    """
Based on the evaluation results, I recommend deploying either Logistic Regression or XGBoost for predicting customer churn in the Telco dataset, as they demonstrated the highest predictive performance.
Further refinement of models and feature engineering could potentially improve predictive accuracy and generalization to unseen data.
Overall, the analysis provides valuable insights into predicting customer churn in the telecommunications industry, which can help businesses make informed decisions to retain customers and improve customer satisfaction.
"""
)
