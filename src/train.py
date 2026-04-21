"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost models.
Evaluates each on the held-out test set, selects the best by ROC-AUC,
and saves it to models/best_model.pkl.

Usage:
    python src/train.py
"""

import os
import sys
import joblib
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import preprocess
from src.utils import evaluate_model, save_model, save_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed. Skipping XGBoost model.")


def get_models() -> dict:
    """Return a dict of {model_name: estimator}."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )

    return models


def train():
    """Main training workflow."""
    print("\n" + "=" * 60)
    print("  LOAN APPROVAL PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # ── 1. Preprocess data ────────────────────────────────────────
    X_train, X_test, y_train, y_test, preprocessor = preprocess(
        filepath="data/train.csv"
    )

    # Transform using the fitted preprocessor
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── 2. Train & evaluate each model ───────────────────────────
    models      = get_models()
    all_metrics = []
    best_score  = -1
    best_name   = None
    best_model  = None

    for name, model in models.items():
        print(f"\n[TRAINING] {name} …")
        model.fit(X_train_t, y_train)

        metrics = evaluate_model(name, model, X_test_t, y_test)
        all_metrics.append(metrics)

        score = metrics["roc_auc"] if metrics["roc_auc"] else metrics["accuracy"]
        if score > best_score:
            best_score = score
            best_name  = name
            best_model = model

    # ── 3. Save best model ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅  Best Model : {best_name}  (ROC-AUC / Accuracy = {best_score:.4f})")
    print(f"{'='*60}")

    os.makedirs("models", exist_ok=True)
    save_model(best_model, "models/best_model.pkl")

    # Save a small metadata file so the API knows which model was chosen
    import json
    meta = {"best_model": best_name, "score": best_score}
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Persist all metrics for analysis
    save_metrics(all_metrics, "models/metrics.json")

    print("\n[DONE] Training complete. Files saved in /models/")


if __name__ == "__main__":
    train()
