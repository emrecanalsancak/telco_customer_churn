import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import RocCurveDisplay


pd.set_option("display.max_columns", None)


df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df.isnull().sum()

df.loc[(df["TotalCharges"] == " "), "TotalCharges"] = np.nan
df.loc[(df["TotalCharges"] == " "), "TotalCharges"]

df["TotalCharges"] = df["TotalCharges"].astype("float64")
df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

df.info()


# Getting numerical and categorical variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identify and categorize columns in a DataFrame based on their data types and cardinality.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        cat_th : int, optional
            Numerical threshold for considering a numerical variable as categorical. Default is 10.
        car_th : int, optional
            Categorical threshold for considering a categorical variable as cardinal. Default is 20.

    Returns
    ------
        cat_cols : list
            List of categorical variables.
        num_cols : list
            List of numerical variables.
        cat_but_car : list
            List of categorical variables with cardinality exceeding 'car_th'.
        num_but_cat : list
            List of numerical variables with cardinality below 'cat_th'.

    Examples
    ------
        # Example: Categorize and display column information for a DataFrame
        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=8, car_th=15)
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
df.head()
cat_cols

yes_or_no_vars = [col for col in df.columns if set(df[col].unique()) == {"Yes", "No"}]

df[yes_or_no_vars] = df[yes_or_no_vars].apply(lambda x: x.map({"Yes": 1, "No": 0}))
df[yes_or_no_vars]

df.info()


def cat_summary(dataframe, col_name, plot=False):
    """
    Display a summary for a categorical column in a DataFrame, including value counts and ratios.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the categorical column for which the summary is generated.
        plot : bool, optional
            Whether to plot a countplot for the categorical column. Default is False.

    Returns
    ------
        None

    Prints a DataFrame with value counts and ratios for each category in the specified categorical column.
    If 'plot' is True, also displays a countplot for the categorical column.

    Examples
    ------
        # Example 1: Display summary for a categorical column
        cat_summary(df, 'categorical_column')

        # Example 2: Display summary for multiple categorical columns
        for col in cat_cols:
            cat_summary(df, col)

        # Example 3: Display summary and plot countplot for a categorical column
        cat_summary(df, 'categorical_column', plot=True)
    """
    print(
        pd.DataFrame(
            {
                f"{col_name}_Count": dataframe[col_name].value_counts(),
                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


for col in cat_cols:
    print(df.groupby(["Churn", col]).size().unstack())


# Outliers
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Calculate the lower and upper bounds for identifying outliers in a numerical column.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the numerical column for which outlier thresholds will be calculated.
        q1 : float, optional
            The lower quartile value. Default is 0.10.
        q3 : float, optional
            The upper quartile value. Default is 0.90.

    Returns
    ------
        low_limit : float
            The lower threshold for identifying outliers.
        up_limit : float
            The upper threshold for identifying outliers.

    Examples
    ------
        # Example: Calculate outlier thresholds for a numerical column
        low_limit, up_limit = outlier_thresholds(df, 'numeric_column')
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Check for outliers in a numerical column of a DataFrame based on custom quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the numerical column to check for outliers.
        q1 : float, optional
            The lower quantile value for calculating the lower threshold. Default is 0.10.
        q3 : float, optional
            The upper quantile value for calculating the upper threshold. Default is 0.90.

    Returns
    ------
        str
            A string indicating the presence of outliers in the specified numerical column.

    Examples
    ------
        # Example: Check for outliers in a numerical column with custom quantiles
        result = check_outlier(df, 'numeric_column', q1=0.05, q3=0.95)
        print(result)

        # Example 2: Check for outliers for multiple numerical columns
        for col in num_cols:
            print(check_outlier(df, col))
    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    print(f"{col_name} Lower Limit: {low_limit}, {col_name} Upper Limit: {up_limit}")

    outliers = dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ]

    num_outliers = outliers.shape[0]  # Count the number of outliers

    if num_outliers > 0:
        return f"{col_name} : {num_outliers} : True"
    else:
        return f"{col_name} : {num_outliers} : False"


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    sns.scatterplot(data=df[col])
    plt.show()


# Creating new features.
df["tenure"].min()
df["Contract"].value_counts()
df["OnlineSecurity"].value_counts()
df["PhoneService"].value_counts()
df["InternetService"].value_counts()
df.info()


# People who have both internet and phone service
df.loc[
    (df["PhoneService"] == 1) & (df["InternetService"] != "No"),
    "Internet_Phone_Service",
] = 1
df.loc[
    (df["PhoneService"] == 0) | (df["InternetService"] == "No"),
    "Internet_Phone_Service",
] = 0

# People who have InternetService but don't have a PhoneService
df["Phone_But_No_Internet"] = 0
df.loc[
    (df["PhoneService"] == 1) & (df["InternetService"] == "No"), "Phone_But_No_Internet"
] = 1


# People who have PhoneService but don't have an InternetService
df["Internet_But_No_Phone"] = 0
df.loc[
    (df["PhoneService"] == 0) & (df["InternetService"] != "No"), "Internet_But_No_Phone"
] = 1


# Based on Tenur grouping the customers as short_term, mid_term, long_term
short_term = df["tenure"].quantile(0.33)
mid_term = df["tenure"].quantile(0.66)

df.loc[(df["tenure"] <= short_term), "TenureGroup"] = "Short_Term"
df.loc[(df["tenure"] > short_term) & (df["tenure"] <= mid_term), "TenureGroup"] = (
    "Mid_Term"
)
df.loc[(df["tenure"] > mid_term), "TenureGroup"] = "Long_Term"


# Grouping customers based on seniority and gender
df.loc[(df["gender"] == "Male") & (df["SeniorCitizen"] == 1), "AgeGenderGroup"] = (
    "SeniorMale"
)
df.loc[(df["gender"] == "Female") & (df["SeniorCitizen"] == 1), "AgeGenderGroup"] = (
    "SeniorFemale"
)
df.loc[(df["gender"] == "Male") & (df["SeniorCitizen"] == 0), "AgeGenderGroup"] = (
    "YoungMale"
)
df.loc[(df["gender"] == "Female") & (df["SeniorCitizen"] == 0), "AgeGenderGroup"] = (
    "YoungFemale"
)
df["AgeGenderGroup"].value_counts()

# Customers with both OnlineSecurity and OnlineBackup
df["OnlineSecurityAndBackup"] = 0
df.loc[
    (df["InternetService"] != "No")
    & (df["OnlineSecurity"] == "Yes")
    & (df["OnlineBackup"] == "Yes"),
    "OnlineSecurityAndBackup",
] = 1
df["OnlineSecurityAndBackup"].value_counts()


# Customers with InternetService based on their AgeGenderGroup
df["SeniorMaleWithInternet"] = 0
df["SeniorFemaleWithInternet"] = 0
df["YoungMaleWithInternet"] = 0
df["YoungFemaleWithInternet"] = 0
df.loc[
    (df["AgeGenderGroup"] == "SeniorMale") & (df["InternetService"] != "No"),
    "SeniorMaleWithInternet",
] = 1
df.loc[
    (df["AgeGenderGroup"] == "SeniorFemale") & (df["InternetService"] != "No"),
    "SeniorFemaleWithInternet",
] = 1
df.loc[
    (df["AgeGenderGroup"] == "YoungMale") & (df["InternetService"] != "No"),
    "YoungMaleWithInternet",
] = 1
df.loc[
    (df["AgeGenderGroup"] == "YoungFemale") & (df["InternetService"] != "No"),
    "YoungFemaleWithInternet",
] = 1


## Splitting the data
train_df, test_df = train_test_split(df, test_size=0.2)

# One-Hot Encoding
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in "Churn"]

train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True, dtype=int)
test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True, dtype=int)


# StandardScaler
ss = StandardScaler()
train_df[num_cols] = ss.fit_transform(train_df[num_cols])
test_df[num_cols] = ss.fit_transform(test_df[num_cols])

# Train sets
X_train = train_df.drop(["customerID", "Churn"], axis=1)
y_train = train_df["Churn"]

# Test sets
X_test = test_df.drop(["customerID", "Churn"], axis=1)
y_test = test_df["Churn"]


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=150),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "GaussianNB": GaussianNB(),
    "XGBoost": XGBClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
}


def evaluate_models(models, X_train, X_test, y_train, y_test, scoring="accuracy"):
    """
    Evaluate a list of models using cross-validation and provide accuracy scores on a separate test set.

    Parameters:
    - models: List of tuples containing (name, model_instance)
    - X_train: Training feature matrix
    - X_test: Test feature matrix
    - y_train: Training target variable
    - y_test: Test target variable
    - scoring: Scoring metric for cross-validation

    Returns:
    - model_performance: Dictionary containing model names and their cross-validation scores
    - test_scores: Dictionary containing model names and their accuracy scores on the test set
    """

    print("Evaluating Base Models...", end="\n")

    # Make a dictionary to keep model scores
    model_performance = {}
    test_scores = {}

    for name, model in models.items():
        # Cross-validation
        cv_results = cross_validate(model, X_train, y_train, cv=3, scoring=scoring)
        model_performance[name] = round(cv_results["test_score"].mean(), 4)

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred = model.predict(X_test)
        test_scores[name] = accuracy_score(y_test, y_pred)

        # Print results
        print(
            f"{scoring}: {model_performance[name]} (CV) | Accuracy: {test_scores[name]} (Test) - {name}"
        )

    return model_performance, test_scores


import time

# start_time = time.time()

model_perf, model_scores = evaluate_models(
    models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Model training took {execution_time} seconds.")
model_perf, model_scores
# accuracy: 0.8033 (CV) | Accuracy: 0.8140525195173882 (Test) - Logistic Regression
# accuracy: 0.7616 (CV) | Accuracy: 0.7757274662881476 (Test) - KNN
# accuracy: 0.7838 (CV) | Accuracy: 0.7984386089425124 (Test) - Random Forest
# accuracy: 0.6768 (CV) | Accuracy: 0.6863023420865862 (Test) - GaussianNB
# accuracy: 0.7758 (CV) | Accuracy: 0.7955997161107168 (Test) - XGBoost
# accuracy: 0.7199 (CV) | Accuracy: 0.7324343506032647 (Test) - DecisionTreeClassifier

# Visualizing the model scores
model_compare = pd.DataFrame(model_scores, index=["Accuracy"])
# model_compare.T.plot.bar()
# plt.show()

# Plotting with custom colors and styles
colors = ["#FF5733", "#C70039", "#900C3F", "#581845", "#073B4C", "#004445"]

# Setting dark background style
plt.style.use("dark_background")

# Plotting
ax = model_compare.T.plot(kind="bar", legend=False)

# Apply custom colors
for i, bar in enumerate(ax.patches):
    bar.set_color(colors[i % len(colors)])

plt.title("Model Comparison", fontsize=16)
plt.xlabel("Models", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(rotation=0, ha="center", fontsize=12)
plt.yticks(fontsize=12)

# Adding data labels
for i, val in enumerate(model_scores.values()):
    plt.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=10)

# plt.tight_layout()

# Saving and showing the plot
# plt.savefig("model_comparison.png")
plt.show()


##########################################################################
# HYPERPARAMETER TUNING
# CONFUSION MATRIX
# CROSS-VALIDATION
# CLASSIFICATION REPORT
# FEATURE IMPORTANCE
# MODEL COMPARISON
##########################################################################

train_scores = []
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 51)

# Setup KNN instance
knn = KNeighborsClassifier()
knn.get_params()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)

    # Fit the algorithm.
    knn.fit(X_train, y_train)

    # Update the training scores list
    train_scores.append(knn.score(X_train, y_train))

    # Update the test scores list
    test_scores.append(knn.score(X_test, y_test))

train_scores
test_scores

plt.plot(neighbors, train_scores, label="Train score", color="#FF5733")  # Red color
plt.plot(neighbors, test_scores, label="Test score", color="#C70039")  # Maroon color
plt.xticks(np.arange(1, 51, 1))
plt.xlabel("Number of neighbors", color="white")  # Setting x-axis label color to white
plt.ylabel("Model score", color="white")  # Setting y-axis label color to white
plt.legend()
plt.title("KNN Hyperparameter Tuning", color="white")  # Setting title color to white

# Saving and showing the plot
# plt.savefig("knn_hyperparam.png", dpi=300)
plt.show()

print(f"Maximum KNN score on the test data: {max(test_scores) * 100:.2f}%")
# After doing hyperparameter tuning on KNN the max result we got is %80.70 Accuracy.


# Tuning KNN with gridSearch
neighbors = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn, neighbors, cv=5, n_jobs=-1, verbose=1).fit(
    X_train, y_train
)

knn_gs_best.best_params_
# Best n_neighbors parameter is 29

knn_final = knn.set_params(**knn_gs_best.best_params_).fit(X_train, y_train)

cv_results = cross_validate(
    knn_final, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

# Accuracy: 0.80
# F1      : 0.58
# Roc_Auc : 0.82

######################################
# Logistic Regression
######################################

log_res = LogisticRegression(max_iter=500)
log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}

gs_log_reg = GridSearchCV(
    log_res,
    log_reg_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# Fit random hyperparameter search model for Logisticregression
gs_log_reg.fit(X_train, y_train)
gs_log_reg.best_params_
gs_log_reg.score(X_test, y_test)

final_log_res = log_res.set_params(**gs_log_reg.best_params_).fit(X_train, y_train)
log_res_cv_results = cross_validate(
    final_log_res, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)

log_res_cv_results["test_accuracy"].mean()
log_res_cv_results["test_f1"].mean()
log_res_cv_results["test_roc_auc"].mean()
# Accuracy: 0.81
# F1      : 0.62
# Roc_Auc : 0.85


#########################################
# Random Forest Classifier
#########################################
ran_fc = RandomForestClassifier(n_jobs=-1)

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
}

rs_rf = RandomizedSearchCV(
    ran_fc, param_distributions=rf_grid, cv=10, n_iter=50, verbose=True
)

# Fit random hyperparameter search model for RandomForestClassifier()
start_time = time.time()

rs_rf.fit(X_train, y_train)

end_time = time.time()
execution_time = end_time - start_time
print(f"Model training took {execution_time} seconds.")
# Find the best hyperparameters.
rs_rf.best_params_

# Evaluate the randomized search for RandomForestClassifier model
rs_rf.score(X_test, y_test)

rf_final = (
    RandomForestClassifier().set_params(**rs_rf.best_params_).fit(X_train, y_train)
)
rf_final.score(X_test, y_test)
# RandomForest accuracy after hyperparameter tuning: 0.82 it was 0.79 before.

rf_cv = cross_validate(
    rf_final, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
rf_cv["test_accuracy"].mean()  # 0.80
rf_cv["test_f1"].mean()  # 0.56
rf_cv["test_roc_auc"].mean()  # 0.84


#############################################
# XGBoost
#############################################
xgb = XGBClassifier()
xgb.get_params()

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.001, 0.01, 0.1, 0.2],
    "max_depth": [3, 4, 5],
    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
}

xgb_random_search = RandomizedSearchCV(
    xgb, param_grid, cv=10, n_iter=50, scoring="accuracy"
)

start_time = time.time()

xgb_random_search.fit(X_train, y_train)

end_time = time.time()
execution_time = end_time - start_time
print(f"Model training took {execution_time} seconds.")

xgb_random_search.best_params_
xgb_random_search.score(X_test, y_test)


xgb_final = (
    XGBClassifier().set_params(**xgb_random_search.best_params_).fit(X_train, y_train)
)

# After hyperparameter tuning XGboost accuracy is 0.81

xgb_cv = cross_validate(
    xgb_final, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
xgb_cv["test_accuracy"].mean()  # 0.80
xgb_cv["test_f1"].mean()  # 0.57
xgb_cv["test_roc_auc"].mean()  # 0.84

#########################################
# Visualization
#########################################
y_preds_knn = knn_final.predict(X_test)
y_preds_log_res = final_log_res.predict(X_test)
y_preds_rf = rf_final.predict(X_test)
y_preds_xgb = xgb_final.predict(X_test)

RocCurveDisplay.from_estimator(knn_final, X_test, y_test)
plt.title("KNN Roc Curve", fontsize=16)
plt.xlabel("True Positive Rate (Positive label: 1)", fontsize=14)
plt.ylabel("False Positive Rate (Positive label: 1)", fontsize=14)
plt.show()

RocCurveDisplay.from_estimator(final_log_res, X_test, y_test)
plt.title("Logistic Regression Roc Curve", fontsize=16)
plt.xlabel("True Positive Rate (Positive label: 1)", fontsize=14)
plt.ylabel("False Positive Rate (Positive label: 1)", fontsize=14)
plt.show()

RocCurveDisplay.from_estimator(rf_final, X_test, y_test)
plt.title("Random Forest Roc Curve", fontsize=16)
plt.xlabel("True Positive Rate (Positive label: 1)", fontsize=14)
plt.ylabel("False Positive Rate (Positive label: 1)", fontsize=14)
plt.show()

RocCurveDisplay.from_estimator(xgb_final, X_test, y_test)
plt.title("XGBoost Roc Curve", fontsize=16)
plt.xlabel("True Positive Rate (Positive label: 1)", fontsize=14)
plt.ylabel("False Positive Rate (Positive label: 1)", fontsize=14)
plt.show()


def plot_confusion_matrix(y, y_pred, model_name):
    """
    Plot a confusion matrix for the predicted and true labels.

    Parameters
    ----------
    y : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Name of the model.

    Returns
    -------
    None

    Examples
    --------
    # Example: Plot confusion matrix for true labels 'y_true' and predicted labels 'y_pred'
    plot_confusion_matrix(y_true, y_pred, model_name="MyModel")
    """

    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.title(f"{model_name} Accuracy Score: {acc}", fontsize=16)
    plt.show()


plot_confusion_matrix(y_test, y_preds_knn, "KNN")
plot_confusion_matrix(y_test, y_preds_log_res, "Logistic Regression")
plot_confusion_matrix(y_test, y_preds_rf, "Random Forest")
plot_confusion_matrix(y_test, y_preds_xgb, "XGBoost")


# Classification report
print(classification_report(y_test, y_preds_knn))
#               precision    recall  f1-score   support

#            0       0.84      0.89      0.86      1030
#            1       0.64      0.55      0.59       379

#     accuracy                           0.80      1409
#    macro avg       0.74      0.72      0.73      1409
# weighted avg       0.79      0.80      0.79      1409

print(classification_report(y_test, y_preds_log_res))
#               precision    recall  f1-score   support

#            0       0.84      0.92      0.88      1030
#            1       0.71      0.53      0.61       379

#     accuracy                           0.82      1409
#    macro avg       0.78      0.73      0.74      1409
# weighted avg       0.81      0.82      0.81      1409

print(classification_report(y_test, y_preds_rf))
#               precision    recall  f1-score   support

#            0       0.83      0.93      0.88      1030
#            1       0.72      0.50      0.59       379

#     accuracy                           0.81      1409
#    macro avg       0.78      0.71      0.73      1409
# weighted avg       0.80      0.81      0.80      1409


print(classification_report(y_test, y_preds_xgb))
#               precision    recall  f1-score   support

#            0       0.84      0.93      0.88      1030
#            1       0.74      0.52      0.61       379

#     accuracy                           0.82      1409
#    macro avg       0.79      0.73      0.75      1409
# weighted avg       0.81      0.82      0.81      1409


############################################################################

######################
# LogisticRegression Feature Importance
######################

# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(final_log_res.coef_[0])))

# Convert dictionary to DataFrame for easier manipulation
feature_df = pd.DataFrame(feature_dict, index=[0])

# Sort features based on their coefficients
feature_df = feature_df.T.sort_values(by=0, ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
colors = plt.cm.autumn(np.linspace(0, 1, len(feature_df)))  # Using a color palette
plt.bar(feature_df.index, feature_df[0], color=colors, width=0.8)
plt.title("Logistic Regression Feature Importance", fontsize=16)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Coefficient", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


##############################
# RandomForest and XGBoost Feature Importance
##############################
plt.style.use("default")


def plot_importance(model, features, num=None, save=False):
    """
    Plot feature importances of a model.

    Parameters
    ----------
    model : object
        The trained model with a `feature_importances_` attribute.
    features : pd.DataFrame
        The DataFrame containing the features used in the model.
    num : int, optional
        Number of top features to display. Default is the total number of features.
    save : bool, optional
        Whether to save the plot as "importances.png". Default is False.

    Returns
    -------
    None

    Examples
    --------
    # Example: Plot feature importances for a RandomForestClassifier
    plot_importance(rf_model, X_train, save=True)
    """

    if num is None:
        num = len(features.columns)

    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )

    # Sort feature importance values
    feature_imp = feature_imp.sort_values(by="Value", ascending=False)[:num]

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp, palette="autumn")
    plt.title("Feature Importances", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig("importances.png", dpi=300)

    plt.show()


plot_importance(rf_final, X_train)
plot_importance(xgb_final, X_train)


##########################################
# Final model comparison
##########################################

models = {
    "KNN": knn_final,
    "Logistic Regression": final_log_res,
    "RandomForestClassifier": rf_final,
    "XGBoost": xgb_final,
}

final_model_scores = {}
for name, model in models.items():
    accuracy_scores = cross_val_score(model, X_test, y_test, cv=10, scoring="accuracy")
    f1_scores = cross_val_score(model, X_test, y_test, cv=10, scoring="f1")
    roc_auc_scores = cross_val_score(model, X_test, y_test, cv=10, scoring="roc_auc")

    final_model_scores[name] = {
        "Accuracy": accuracy_scores.mean(),
        "F1 Score": f1_scores.mean(),
        "ROC AUC": roc_auc_scores.mean(),
    }

scores_df = pd.DataFrame(final_model_scores)

# Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")  # Set style
colors = sns.color_palette("husl", len(scores_df))  # Choose color palette
scores_df.T.plot(kind="bar", color=colors)

# Add labels and title
plt.title("Model Performance Comparison", fontsize=16)
plt.xlabel("Metrics", fontsize=14)
plt.ylabel("Scores", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Models", bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.show()
