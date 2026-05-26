
# =========================================================
# GCI Homework 5: Model Evaluation with Cross-Validation
# =========================================================

# -----------------------------
# 1. Imports
# -----------------------------
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# -----------------------------
# 2. Load Dataset
# -----------------------------
data = load_breast_cancer(as_frame=True)

X = data.frame.drop(columns=['target'])
y = data.frame['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 3. Homework Function (SUBMIT THIS ONLY)
# -----------------------------
def homework(X_train, y_train, model_type, k):

    # Select model
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)

    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)

    elif model_type == 'svm':
        model = SVC()

    else:
        raise ValueError("Invalid model_type")

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=k)

    return scores.mean()


# -----------------------------
# 4. TESTING SECTION (LOCAL ONLY)
# -----------------------------
if __name__ == "__main__":

    print("=== Testing Logistic Regression ===")
    print(homework(X_train, y_train, 'logistic_regression', 5))

    print("\n=== Testing Decision Tree ===")
    print(homework(X_train, y_train, 'decision_tree', 5))

    print("\n=== Testing SVM ===")
    print(homework(X_train, y_train, 'svm', 5))