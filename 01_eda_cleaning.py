"""
Day 2: EDA + Data Cleaning
UCI Default of Credit Card Clients Dataset

Run this top to bottom. It prints findings at each step so you can see
exactly what's wrong with the raw data and why each cleaning decision
is made - this is what you explain in an interview, not just the code.
"""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------
df = pd.read_csv("data/raw/credit_default.csv")
print("Raw shape:", df.shape)
print()

# ---------------------------------------------------------------------
# 2. First look
# ---------------------------------------------------------------------
print("Missing values per column:")
print(df.isnull().sum().sum(), "total nulls (should be 0 - this dataset has none)")
print()

print("Target distribution (default.payment.next.month):")
print(df["default.payment.next.month"].value_counts(normalize=True))
print("-> ~22% defaulters. This is an IMBALANCED dataset. Accuracy alone")
print("   is a misleading metric here - a model that always predicts 'no")
print("   default' already scores ~78% accuracy while being useless.")
print()

# ---------------------------------------------------------------------
# 3. Known data-quality issues in this dataset (documented anomalies)
# ---------------------------------------------------------------------
print("EDUCATION value counts (1=grad school,2=university,3=high school,4=other):")
print(df["EDUCATION"].value_counts().sort_index())
print("-> categories 0, 5, 6 are UNDOCUMENTED. Treat as 'other/unknown'.")
print()

print("MARRIAGE value counts (1=married,2=single,3=other):")
print(df["MARRIAGE"].value_counts().sort_index())
print("-> category 0 is UNDOCUMENTED. Treat as 'other/unknown'.")
print()

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
print("PAY_n columns min/max (repayment status):")
print(df[pay_cols].describe().loc[["min", "max"]])
print("-> documented scale is -1 (pay duly) to 9 (9+ months late).")
print("   -2 appears and is UNDOCUMENTED (likely 'no consumption'/no balance).")
print("   We keep it as its own category rather than dropping rows, since")
print("   it's ~a fifth of the -1/-2 values combined and dropping would")
print("   meaningfully shrink the dataset.")
print()

# ---------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------
clean = df.copy()

# Rename PAY_0 -> PAY_1 for consistency with PAY_2..PAY_6 (documented UCI quirk)
clean = clean.rename(columns={"PAY_0": "PAY_1"})

# Collapse undocumented categories into an explicit "other" bucket (4)
# rather than deleting rows - deleting here would drop ~468 rows (~1.6%)
# non-randomly (these aren't missing-at-random), which risks bias.
clean["EDUCATION"] = clean["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
clean["MARRIAGE"] = clean["MARRIAGE"].replace({0: 3})

# Drop ID - it's a row identifier, not a feature, and leaking it in would
# let models "memorize" rows instead of learning the true pattern.
clean = clean.drop(columns=["ID"])

# Rename target to something code-friendly
clean = clean.rename(columns={"default.payment.next.month": "target"})

print("After cleaning:")
print("EDUCATION:", sorted(clean["EDUCATION"].unique()))
print("MARRIAGE:", sorted(clean["MARRIAGE"].unique()))
print("Shape:", clean.shape)
print()

# ---------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------
clean.to_csv("data/processed/credit_default_clean.csv", index=False)
print("Saved cleaned data to data/processed/credit_default_clean.csv")
