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
from sklearn.metrics import confusion_matrix, classification_report
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

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.head()
cat_cols

yes_or_no_vars = [col for col in df.columns if set(df[col].unique()) == {"Yes", "No"}]

df[yes_or_no_vars] = df[yes_or_no_vars].apply(lambda x: x.map({"Yes": 1, "No": 0}))
df[yes_or_no_vars]

df.info()


def cat_summary(dataframe, col_name, plot=False):
    print(
        pd.DataFrame(
            {
                col_name: dataframe[col_name].value_counts(),
                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )
    print("#####################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


for col in cat_cols:
    print(df.groupby(["Churn", col]).size().unstack())


# Outliers
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    non_nan_values = dataframe[col_name].dropna()  # Exclude NaN values
    if ((non_nan_values > up_limit) | (non_nan_values < low_limit)).any():
        return True
    else:
        return False


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
df.loc[
    (df["tenure"] > short_term) & (df["tenure"] <= mid_term), "TenureGroup"
] = "Mid_Term"
df.loc[(df["tenure"] > mid_term), "TenureGroup"] = "Long_Term"


# Grouping customers based on seniority and gender
df.loc[
    (df["gender"] == "Male") & (df["SeniorCitizen"] == 1), "AgeGenderGroup"
] = "SeniorMale"
df.loc[
    (df["gender"] == "Female") & (df["SeniorCitizen"] == 1), "AgeGenderGroup"
] = "SeniorFemale"
df.loc[
    (df["gender"] == "Male") & (df["SeniorCitizen"] == 0), "AgeGenderGroup"
] = "YoungMale"
df.loc[
    (df["gender"] == "Female") & (df["SeniorCitizen"] == 0), "AgeGenderGroup"
] = "YoungFemale"
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
cat_cols, num_cols, cat_but_car = grab_col_names(df)
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


def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.

    Parameters
    ----------
    models : dict
        A dict of different Scikit-Learn machine learning models.
    X_train : DataFrame
        Training data (no labels)
    X_test : DataFrame
        Testing data (no labels)
    y_train : Pandas Series
        Training labels
    y_test : Pandas Series
        Test labels
    """

    # Make a dictionary to keep model scores.
    model_scores = {}

    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)

        # Evaluate the model and append its scores to model_scores
        model_scores[name] = model.score(X_test, y_test)

    return model_scores


import time

# df.isnull().sum()
# start_time = time.time()

model_scores = fit_and_score(
    models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Model training took {execution_time} seconds.")
model_scores

# Visualizing the model scores
model_compare = pd.DataFrame(model_scores, index=["Accuracy"])
model_compare.T.plot.bar()
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

plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 51, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()
plt.show()

print(f"Maximum KNN score on the test data: {max(test_scores) * 100:.2f}%")
# After doing hyperparameter tuning on KNN the max result we got is %80.84 Accuracy.


# Tuning KNN with gridSearch
neighbors = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn, neighbors, cv=10, n_jobs=-1, verbose=1).fit(
    X_train, y_train
)

knn_gs_best.best_params_
# Best n_neighbors parameter is 46

knn_final = knn.set_params(**knn_gs_best.best_params_).fit(X_train, y_train)

cv_results = cross_validate(
    knn_final, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

# Accuracy: 0.81
# F1      : 0.58
# Roc_Auc : 0.81

######################################
# Logistic Regression
######################################

# Before tuning the accuracy of logres was: 0.82
log_res = LogisticRegression(max_iter=150)
# log_res.get_params()

log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}


rs_log_reg = RandomizedSearchCV(
    log_res,
    param_distributions=log_reg_grid,
    cv=10,
    n_iter=20,
    verbose=True,
)

# Fit random hyperparameter search model for Logisticregression
rs_log_reg.fit(X_train, y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)

final_log_res = log_res.set_params(**rs_log_reg.best_params_).fit(X_train, y_train)
log_res_cv_results = cross_validate(
    final_log_res, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
log_res_cv_results = cross_validate(
    final_log_res, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)

log_res_cv_results["test_accuracy"].mean()
log_res_cv_results["test_f1"].mean()
log_res_cv_results["test_roc_auc"].mean()

# gs_log_reg = GridSearchCV(log_res, param_grid=log_reg_grid, cv=10, verbose=True)

# # Fit grid hyperparameter search model
# gs_log_reg.fit(X_train, y_train)

# gs_log_reg.best_params_

# # Evaluate the grid search Logistic Regression model
# gs_log_reg.score(X_test, y_test)

# gs_log_res_results = cross_validate(
#     rs_log_reg, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
# )

# gs_log_res_results["test_accuracy"].mean()
# gs_log_res_results["test_f1"].mean()
# gs_log_res_results["test_roc_auc"].mean()

# grid_final_log_res = log_res.set_params(**gs_log_reg.best_params_).fit(X_train, y_train)
# grid_log_res_cv_results = cross_validate(
#     grid_final_log_res, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
# )

# grid_log_res_cv_results["test_accuracy"].mean()
# grid_log_res_cv_results["test_f1"].mean()
# grid_log_res_cv_results["test_roc_auc"].mean()

# RandomizedSearch and GridSearchCV almost identical
# Accuracy: 0.81
# F1      : 0.58
# Roc_Auc : 0.84


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
    ran_fc, param_distributions=rf_grid, cv=10, n_iter=20, verbose=True
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
rf_cv["test_accuracy"].mean()  # 0.81
rf_cv["test_f1"].mean()  # 0.55
rf_cv["test_roc_auc"].mean()  # 0.83


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

xgb_grid_search = GridSearchCV(xgb, param_grid, cv=10, scoring="accuracy")

start_time = time.time()

xgb_grid_search.fit(X_train, y_train)

end_time = time.time()
execution_time = end_time - start_time
print(f"Model training took {execution_time} seconds.")

xgb_grid_search.best_params_
xgb_grid_search.score(X_test, y_test)


xgb_final = (
    XGBClassifier().set_params(**xgb_grid_search.best_params_).fit(X_train, y_train)
)

# After hyperparameter tuning XGboost accuracy is 0.81

xgb_cv = cross_validate(
    xgb_final, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
xgb_cv["test_accuracy"].mean()  # 0.81
xgb_cv["test_f1"].mean()  # 0.54
xgb_cv["test_roc_auc"].mean()  # 0.84

#########################################
# Visualization
#########################################
y_preds_knn = knn_final.predict(X_test)
y_preds_log_res = final_log_res.predict(X_test)
y_preds_rf = rf_final.predict(X_test)
y_preds_xgb = xgb_final.predict(X_test)

RocCurveDisplay.from_estimator(knn_final, X_test, y_test)
plt.show()

RocCurveDisplay.from_estimator(final_log_res, X_test, y_test)
plt.show()

RocCurveDisplay.from_estimator(rf_final, X_test, y_test)
plt.show()

RocCurveDisplay.from_estimator(xgb_final, X_test, y_test)
plt.show()


def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice-looking confusion matrix using seaborn's heatmap.
    """
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        confusion_matrix(y_test, y_preds),
        annot=True,
        cbar=False,
        fmt="d",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


plot_conf_mat(y_test, y_preds_knn)
plot_conf_mat(y_test, y_preds_log_res)
plot_conf_mat(y_test, y_preds_rf)
plot_conf_mat(y_test, y_preds_xgb)


# Classification report
print(classification_report(y_test, y_preds_knn))
#               precision    recall  f1-score   support

#            0       0.86      0.89      0.87      1054
#            1       0.63      0.56      0.59       355

#     accuracy                           0.81      1409
#    macro avg       0.74      0.73      0.73      1409
# weighted avg       0.80      0.81      0.80      1409

print(classification_report(y_test, y_preds_log_res))
#               precision    recall  f1-score   support

#            0       0.86      0.90      0.88      1054
#            1       0.65      0.57      0.61       355

#     accuracy                           0.81      1409
#    macro avg       0.76      0.73      0.74      1409
# weighted avg       0.81      0.81      0.81      1409

print(classification_report(y_test, y_preds_rf))
#               precision    recall  f1-score   support

#            0       0.86      0.90      0.88      1054
#            1       0.65      0.57      0.61       355

#     accuracy                           0.81      1409
#    macro avg       0.76      0.73      0.74      1409
# weighted avg       0.81      0.81      0.81      1409


print(classification_report(y_test, y_preds_xgb))
#            0       0.86      0.91      0.88      1054
#            1       0.67      0.55      0.61       355

#     accuracy                           0.82      1409
#    macro avg       0.77      0.73      0.74      1409
# weighted avg       0.81      0.82      0.81      1409


############################################################################

######################
# LogisticRegression Feature Importance
######################

# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(final_log_res.coef_[0])))

# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False)
plt.show()


##############################
# RandomForest and XGBoost Feature Importance
##############################


def plot_features(columns, importances, n=20):
    df = (
        (pd.DataFrame({"features": columns, "feature_importances": importances}))
        .sort_values("feature_importances", ascending=False)
        .reset_index(drop=True)
    )

    # Plot the df we created
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:n])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature Importances")
    ax.invert_yaxis()
    plt.show()


plot_features(X_train.columns, rf_final.feature_importances_)
plot_features(X_train.columns, xgb_final.feature_importances_)


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
scores_df.T.plot.bar()
plt.show()
