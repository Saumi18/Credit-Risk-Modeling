"""
Day 4-6: Baseline Models - Logistic Regression + LightGBM

Reports accuracy, precision, recall, F1, and AUC - not just accuracy,
because this dataset is imbalanced (~22% positive class). A model that
predicts "no default" for everyone scores ~78% accuracy while being
completely useless, so accuracy alone would be a misleading headline
number here.

Scaling is fit ONLY on X_train (Day 3 already split before this step),
then applied to X_test - this is the leakage-safe order.
"""
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------
# 1. Load the splits from Day 3 - never re-split here, always reuse
# ---------------------------------------------------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------------------------------------------------------------------
# 2. Scale - fit on train ONLY, then transform both
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 3. Helper to evaluate honestly and consistently across models
# ---------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_proba), 4),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))
    return metrics

results = []

# ---------------------------------------------------------------------
# 4. Baseline 1: Logistic Regression with class_weight balanced
#    (imbalance-aware from the start, since plain LR would just predict
#    the majority class most of the time otherwise)
# ---------------------------------------------------------------------
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("LogisticRegression", y_test, lr_pred, lr_proba))

# ---------------------------------------------------------------------
# 5. Baseline 2: LightGBM (handles imbalance via scale_pos_weight)
#    Trees don't need scaled input, but we reuse X_train_scaled for a
#    fair like-for-like comparison at this baseline stage.
# ---------------------------------------------------------------------
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

lgbm = LGBMClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    verbosity=-1,
)
lgbm.fit(X_train_scaled, y_train)
lgbm_pred = lgbm.predict(X_test_scaled)
lgbm_proba = lgbm.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("LightGBM", y_test, lgbm_pred, lgbm_proba))

# ---------------------------------------------------------------------
# 6. Save models + scaler + metrics (model card)
# ---------------------------------------------------------------------
import os
os.makedirs("models", exist_ok=True)
os.makedirs("docs", exist_ok=True)

joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(lgbm, "models/lightgbm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

with open("docs/model_card.md", "w") as f:
    f.write("# Model Card - Credit Risk Baseline Models\n\n")
    f.write("Dataset: UCI Default of Credit Card Clients (30,000 rows, 22.1% positive class)\n\n")
    f.write("Split: 80/20 stratified, random_state=42, scaler fit on train only (no leakage)\n\n")
    f.write("## Results\n\n")
    f.write("| Model | Accuracy | Precision | Recall | F1 | AUC |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in results:
        f.write(f"| {r['model']} | {r['accuracy']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['auc']} |\n")
    f.write("\n## Notes\n\n")
    f.write("- Accuracy alone is misleading here: a model predicting 'no default' for\n")
    f.write("  every row scores ~78% accuracy while being useless. Precision/recall/F1/AUC\n")
    f.write("  are the metrics that actually matter for this imbalanced problem.\n")
    f.write("- Both models use class-imbalance handling (class_weight/scale_pos_weight)\n")
    f.write("  rather than plain defaults, which would otherwise barely predict the\n")
    f.write("  minority (default) class at all.\n")
    f.write("- These are BASELINE, untuned models. Next steps: SMOTE comparison,\n")
    f.write("  hyperparameter tuning, threshold tuning on the precision-recall curve.\n")

with open("docs/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved models to models/, scaler to models/scaler.pkl")
print("Saved docs/model_card.md and docs/metrics.json")
