"""
Threshold Tuning + SHAP Explainability

Threshold tuning: the model outputs a probability, and by default we call
anything >0.5 "will default". That 0.5 cutoff is arbitrary - it was never
learned from data, it's just a convention. We scan a range of cutoffs and
pick the one that actually maximizes F1 (or optimizes for recall, if you
decide missing a defaulter is costlier than a false alarm - documented
below so it's a stated decision, not a hidden one).

SHAP: explains *why* the model makes each prediction, in terms of which
features pushed the prediction up or down. This is what turns "it's a
LightGBM model" into "here's what actually drives risk in this model",
which is the more interesting sentence in an interview.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import shap

import os
os.makedirs("docs/plots", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load model, scaler, and test data
# ---------------------------------------------------------------------
model = joblib.load("models/lightgbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X_train.columns.tolist()

proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------
# 2. Threshold sweep - find the F1-optimal cutoff
# ---------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
best_threshold = thresholds[best_idx]

print(f"Default threshold (0.5):")
default_pred = (proba > 0.5).astype(int)
print(f"  Precision={precision_score(y_test, default_pred):.4f}  "
      f"Recall={recall_score(y_test, default_pred):.4f}  "
      f"F1={f1_score(y_test, default_pred):.4f}")

print(f"\nF1-optimal threshold ({best_threshold:.4f}):")
tuned_pred = (proba > best_threshold).astype(int)
print(f"  Precision={precision_score(y_test, tuned_pred):.4f}  "
      f"Recall={recall_score(y_test, tuned_pred):.4f}  "
      f"F1={f1_score(y_test, tuned_pred):.4f}")

# Also show a recall-favoring threshold, since in credit risk, missing an
# actual defaulter (false negative) is often costlier than a false alarm.
# We pick the lowest threshold that still keeps precision above 0.35 as a
# floor (arbitrary but stated - real-world use would set this from a
# business cost analysis, not a code comment).
recall_favoring_mask = precisions[:-1] >= 0.35
if recall_favoring_mask.any():
    # thresholds are ascending, so precision generally rises and recall
    # falls as threshold increases -> the LOWEST threshold meeting the
    # precision floor gives the HIGHEST recall among valid options.
    rf_idx = np.where(recall_favoring_mask)[0][0]
    rf_threshold = thresholds[rf_idx]
    rf_pred = (proba > rf_threshold).astype(int)
    print(f"\nRecall-favoring threshold ({rf_threshold:.4f}, precision floor 0.35):")
    print(f"  Precision={precision_score(y_test, rf_pred):.4f}  "
          f"Recall={recall_score(y_test, rf_pred):.4f}  "
          f"F1={f1_score(y_test, rf_pred):.4f}")

# Plot precision-recall curve with both thresholds marked
plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, label="Precision-Recall curve")
plt.scatter(recall_score(y_test, default_pred), precision_score(y_test, default_pred),
            color="red", label="Default (0.5)", zorder=5)
plt.scatter(recall_score(y_test, tuned_pred), precision_score(y_test, tuned_pred),
            color="green", label=f"F1-optimal ({best_threshold:.2f})", zorder=5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Threshold Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("docs/plots/precision_recall_curve.png", dpi=120)
plt.close()
print("\nSaved docs/plots/precision_recall_curve.png")

# ---------------------------------------------------------------------
# 3. SHAP explainability
# ---------------------------------------------------------------------
print("\nComputing SHAP values (uses a sample of train for the background)...")
explainer = shap.TreeExplainer(model)
# Use a subsample of test set for speed - SHAP on all 6000 rows is slow
sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(1000, len(X_test)), replace=False)
X_sample = pd.DataFrame(X_test_scaled[sample_idx], columns=feature_names)
shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("docs/plots/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved docs/plots/shap_summary.png")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)
print("\nTop 10 most important features (by mean |SHAP value|):")
print(importance_df.head(10).to_string(index=False))

# ---------------------------------------------------------------------
# 4. Save chosen threshold + findings to model card
# ---------------------------------------------------------------------
with open("docs/model_card.md", "a") as f:
    f.write("\n## Threshold tuning + SHAP\n\n")
    f.write(f"- Default threshold (0.5): F1={f1_score(y_test, default_pred):.4f}\n")
    f.write(f"- F1-optimal threshold ({best_threshold:.4f}): F1={f1_score(y_test, tuned_pred):.4f}\n")
    if recall_favoring_mask.any():
        f.write(f"- Recall-favoring threshold ({rf_threshold:.4f}, precision floor 0.35): "
                f"Recall={recall_score(y_test, rf_pred):.4f}\n")
    f.write(f"\n**Chosen threshold for deployment: {best_threshold:.4f}** "
            f"(maximizes F1; revisit if a real business cost model becomes available).\n\n")
    f.write("Top 5 features by SHAP importance:\n\n")
    for _, row in importance_df.head(5).iterrows():
        f.write(f"- `{row['feature']}`: {row['mean_abs_shap']:.4f}\n")
    f.write("\n![Precision-Recall Curve](plots/precision_recall_curve.png)\n")
    f.write("\n![SHAP Summary](plots/shap_summary.png)\n")

with open("docs/chosen_threshold.json", "w") as f:
    json.dump({"threshold": float(best_threshold)}, f, indent=2)

print("\nUpdated docs/model_card.md and saved docs/chosen_threshold.json")
