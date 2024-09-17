import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For model saving

# Train-test split function


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Function to train multiple models and select the best one


def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
    }

    best_model = None
    best_score = 0
    best_model_name = ''

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    print(f"Best model: {best_model_name} with Accuracy: {best_score:.4f}")
    return best_model, best_model_name

# Function for hyperparameter tuning of the best model


def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring='accuracy', cv=5
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Function for model interpretability using SHAP


def model_interpretability(model, X_test):
    # Use appropriate SHAP explainer based on model type
    if isinstance(
        model,
        (
            RandomForestClassifier,
            DecisionTreeClassifier,
            GradientBoostingClassifier,
        ),
    ):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    else:
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values is a list; take the values for the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, X_test)
    plt.show()

# Function to evaluate the model with multiple metrics


def evaluate_model(y_test, y_pred, model_name, y_proba=None):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nEvaluation Metrics for {model_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"ROC-AUC Score: {roc_auc:.4f}")

# Function to save the model to disk


def save_model(model, filename='best_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

# Full pipeline function: train, tune, evaluate, interpret, and save models


def model_pipeline(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_model, best_model_name = train_models(
        X_train, y_train, X_test, y_test
    )

    # Define parameter grid for hyperparameter tuning if necessary
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
        }
        best_model = tune_hyperparameters(
            best_model, param_grid, X_train, y_train
        )

    y_pred = best_model.predict(X_test)
    y_proba = (
        best_model.predict_proba(X_test)
        if hasattr(best_model, "predict_proba")
        else None
    )

    evaluate_model(y_test, y_pred, best_model_name, y_proba)
    model_interpretability(best_model, X_test)
    save_model(best_model)
