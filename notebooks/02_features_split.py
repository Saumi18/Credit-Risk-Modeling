"""
Day 3: Feature Engineering + Train/Test Split

Order matters here and is the #1 place people accidentally leak data:
  1. split FIRST (on raw/cleaned features)
  2. engineer features using only within-row math (safe - no leakage)
  3. scaling/encoding happens LATER (Day 5+), fit only on the train set

If you fit a scaler or compute any statistic (mean, std, an aggregate)
on the full dataset before splitting, information from the test set
leaks into training and your metrics become optimistic lies.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/credit_default_clean.csv")
print("Loaded:", df.shape)

# ---------------------------------------------------------------------
# 1. Feature engineering (row-wise math only - no leakage risk)
# ---------------------------------------------------------------------
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
pay_status_cols = [f"PAY_{i}" for i in range(1, 7)]

# Utilization ratio: how much of their credit limit they're using on
# average across the 6 months. High utilization is a classic risk signal.
df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
df["utilization_ratio"] = df["avg_bill_amt"] / df["LIMIT_BAL"].replace(0, np.nan)
df["utilization_ratio"] = df["utilization_ratio"].fillna(0).clip(lower=0)

# Payment-to-bill ratio: are they paying back a healthy fraction of what
# they owe, or barely making minimum payments? Low ratio = risk signal.
df["avg_pay_amt"] = df[pay_amt_cols].mean(axis=1)
df["payment_ratio"] = df["avg_pay_amt"] / df["avg_bill_amt"].replace(0, np.nan)
df["payment_ratio"] = df["payment_ratio"].fillna(0).clip(lower=0, upper=5)

# Delay trend: is their repayment status getting WORSE over the 6 months
# (early months minus late months)? Positive = deteriorating behavior.
# PAY_1 is the most recent month, PAY_6 is the oldest.
df["delay_trend"] = df["PAY_1"] - df["PAY_6"]

# Count of months with any late payment (status > 0 means late)
df["months_late"] = (df[pay_status_cols] > 0).sum(axis=1)

# Max delay ever seen across the 6 months - captures worst-case behavior
df["max_delay"] = df[pay_status_cols].max(axis=1)

print("\nNew engineered features:")
print(["avg_bill_amt", "utilization_ratio", "avg_pay_amt", "payment_ratio",
       "delay_trend", "months_late", "max_delay"])

print("\nQuick sanity check - correlation with target:")
new_feats = ["utilization_ratio", "payment_ratio", "delay_trend", "months_late", "max_delay"]
print(df[new_feats + ["target"]].corr()["target"].sort_values(ascending=False))

# ---------------------------------------------------------------------
# 2. Train/test split - stratified because of class imbalance
# ---------------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,       # fixed seed = reproducible split
    stratify=y,            # preserves ~22% default rate in both sets
)

print("\nTrain shape:", X_train.shape, "  Test shape:", X_test.shape)
print("Train target rate:", y_train.mean().round(4))
print("Test target rate:", y_test.mean().round(4))
print("-> both should be ~0.221, confirming stratification worked")

# ---------------------------------------------------------------------
# 3. Save - as separate files so Day 5+ scripts just load these directly
#    and never risk re-splitting differently by accident
# ---------------------------------------------------------------------
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("\nSaved train/test splits to data/processed/")
