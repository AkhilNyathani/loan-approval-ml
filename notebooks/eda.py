"""
notebooks/eda.py
----------------
Exploratory Data Analysis script.
Run this to understand the dataset before training.
(Convert to .ipynb by copying cells into Jupyter if preferred.)

Usage:
    python notebooks/eda.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_PATH = "data/train.csv"


def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at {DATA_PATH}.")
        print("Download from: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n" + "="*60)
    print("  LOAN DATASET — EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\n📋 Columns:\n", list(df.columns))

    print("\n🔢 Data Types:\n", df.dtypes.to_string())

    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found!")
    else:
        print(missing.to_string())

    print("\n📊 Target Distribution (Loan_Status):")
    if "Loan_Status" in df.columns:
        print(df["Loan_Status"].value_counts(normalize=True).mul(100).round(2).to_string())

    print("\n📈 Numerical Summary:")
    print(df.describe().round(2).to_string())

    print("\n🔤 Categorical Summaries:")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        print(f"\n  {col}:")
        print(df[col].value_counts().to_string())

    print("\n[DONE] EDA complete.")


if __name__ == "__main__":
    run_eda()
