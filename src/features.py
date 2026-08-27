"""
src/features.py

The SAME feature engineering logic from notebooks/02_features_split.py,
extracted into a function so the API (Part 3) computes features identically
to how the model was trained - this is the #1 place production ML systems
silently break: notebook and API code drifting out of sync.
"""
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the 23 raw UCI columns, returns all 30 model columns.
    Mirrors notebooks/02_features_split.py exactly."""
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = [f"PAY_{i}" for i in range(1, 7)]

    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["utilization_ratio"] = df["avg_bill_amt"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["utilization_ratio"] = df["utilization_ratio"].fillna(0).clip(lower=0)

    df["avg_pay_amt"] = df[pay_amt_cols].mean(axis=1)
    df["payment_ratio"] = df["avg_pay_amt"] / df["avg_bill_amt"].replace(0, np.nan)
    df["payment_ratio"] = df["payment_ratio"].fillna(0).clip(lower=0, upper=5)

    df["delay_trend"] = df["PAY_1"] - df["PAY_6"]
    df["months_late"] = (df[pay_status_cols] > 0).sum(axis=1)
    df["max_delay"] = df[pay_status_cols].max(axis=1)

    return df


def clean_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors notebooks/01_eda_cleaning.py's undocumented-category fix.
    Apply this to raw incoming data before feature engineering."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df
