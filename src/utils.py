"""
utils.py
--------
Shared helper functions used across the project.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def save_model(model, path: str) -> None:
    """Persist a fitted sklearn-compatible model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved → {path}")


def load_model(path: str):
    """Load a previously saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def evaluate_model(name: str, model, X_test_transformed, y_test) -> dict:
    """
    Evaluate a trained model and print a formatted report.
    Returns a dict of key metrics.
    """
    y_pred = model.predict(X_test_transformed)
    y_prob = (
        model.predict_proba(X_test_transformed)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    report = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])

    print(f"\n{'='*55}")
    print(f"  Model : {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    if roc:
        print(f"  ROC-AUC   : {roc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")

    return {
        "name": name,
        "accuracy": round(acc, 4),
        "roc_auc": round(roc, 4) if roc else None,
        "confusion_matrix": cm.tolist(),
    }


def save_metrics(metrics_list: list, path: str = "models/metrics.json") -> None:
    """Save a list of metric dicts to JSON for later reference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"[INFO] Metrics saved → {path}")


def input_to_dataframe(raw_input: dict) -> pd.DataFrame:
    """
    Convert a raw prediction request dict into a single-row DataFrame
    with the exact column order expected by the preprocessor.
    """
    EXPECTED_COLS = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area",
    ]
    # Normalise '3+' → '3'
    if "Dependents" in raw_input and raw_input["Dependents"] == "3+":
        raw_input["Dependents"] = "3"

    df = pd.DataFrame([raw_input])

    # Ensure all expected columns are present
    for col in EXPECTED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required field: '{col}'")

    return df[EXPECTED_COLS]
