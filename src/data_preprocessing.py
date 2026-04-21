"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, encoding, and splitting.
Run this to produce a fitted preprocessor saved to /models/preprocessor.pkl
and train/test CSV files saved to /data/.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ── Column groups ───────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area"]
NUMERICAL_COLS   = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term", "Credit_History"]
TARGET_COL       = "Loan_Status"
LOAN_ID_COL      = "Loan_ID"


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download the Loan Prediction dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset\n"
            "and place train.csv inside the /data folder."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID column and fix known quirks in the raw dataset."""
    df = df.copy()

    # Drop Loan_ID – it carries no predictive value
    if LOAN_ID_COL in df.columns:
        df.drop(columns=[LOAN_ID_COL], inplace=True)

    # Standardise Dependents: '3+' → '3'
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")

    # Encode target: Y → 1, N → 0
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})

    print("[INFO] Data cleaned.")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
      - Imputes + One-Hot encodes categorical columns
      - Imputes (median) + Standard-scales numerical columns
    """
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("num", numerical_pipeline, NUMERICAL_COLS),
    ])

    return preprocessor


def preprocess(filepath: str = "data/train.csv",
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Full preprocessing pipeline:
      1. Load → 2. Clean → 3. Split → 4. Fit preprocessor on train
      5. Save processed splits + fitted preprocessor
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load & clean
    df = load_data(filepath)
    df = clean_data(df)

    # 2. Separate features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Split → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 4. Fit preprocessor on training data only
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # 5. Persist
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("[INFO] Preprocessor saved → models/preprocessor.pkl")

    # Save splits for reproducibility / notebook exploration
    X_train.assign(Loan_Status=y_train.values).to_csv("data/train_processed.csv", index=False)
    X_test.assign(Loan_Status=y_test.values).to_csv("data/test_processed.csv",  index=False)
    print("[INFO] Processed CSVs saved → data/train_processed.csv, data/test_processed.csv")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    preprocess("data/train.csv")
