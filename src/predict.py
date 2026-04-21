"""
predict.py
----------
Reusable prediction pipeline.
Loads the saved preprocessor + best model and returns a prediction
for any raw applicant dict.

Usage (standalone):
    python src/predict.py
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model, input_to_dataframe


# ── Lazy-loaded singletons (loaded once per process) ────────────────────────
_preprocessor = None
_model        = None


def _load_artifacts():
    """Load preprocessor and model from disk (cached after first call)."""
    global _preprocessor, _model

    if _preprocessor is None:
        _preprocessor = load_model("models/preprocessor.pkl")

    if _model is None:
        _model = load_model("models/best_model.pkl")


def predict(raw_input: dict) -> dict:
    """
    Given a raw applicant dict, return prediction result.

    Parameters
    ----------
    raw_input : dict
        Keys must match the feature set (see utils.input_to_dataframe).

    Returns
    -------
    dict with keys:
        - prediction  : int  (1 = Approved, 0 = Rejected)
        - label       : str  ("Approved" | "Rejected")
        - probability : float (probability of approval, 0-1)
    """
    _load_artifacts()

    # Convert raw dict → DataFrame
    df = input_to_dataframe(raw_input)

    # Apply fitted preprocessor
    X_transformed = _preprocessor.transform(df)

    # Predict
    pred_label = int(_model.predict(X_transformed)[0])

    # Probability (if model supports it)
    if hasattr(_model, "predict_proba"):
        prob = float(_model.predict_proba(X_transformed)[0][1])
    else:
        prob = float(pred_label)

    label = "Approved" if pred_label == 1 else "Rejected"

    return {
        "prediction":  pred_label,
        "label":       label,
        "probability": round(prob, 4),
    }


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "Gender":            "Male",
        "Married":           "Yes",
        "Dependents":        "0",
        "Education":         "Graduate",
        "Self_Employed":     "No",
        "ApplicantIncome":   5000,
        "CoapplicantIncome": 1500,
        "LoanAmount":        120,
        "Loan_Amount_Term":  360,
        "Credit_History":    1,
        "Property_Area":     "Urban",
    }

    result = predict(sample)
    print("\n── Prediction Result ──────────────────────")
    print(f"  Status      : {result['label']}")
    print(f"  Probability : {result['probability']:.2%}")
    print("───────────────────────────────────────────")
