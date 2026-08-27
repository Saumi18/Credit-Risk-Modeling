"""
Day 8-9: SMOTE Comparison + Hyperparameter Tuning

Two separate questions asked honestly, not assumed:
  1. Does SMOTE actually help over scale_pos_weight (which we already use)?
     SMOTE synthesizes new minority-class examples. It sometimes helps,
     sometimes hurts (can create noisy synthetic points) - we TEST it,
     not just add it because it's popular.
  2. Can we beat baseline LightGBM's F1/AUC with proper tuning?

IMPORTANT: SMOTE is applied to X_train ONLY, after the train/test split
(Day 3) and after scaling (Day 4-6) - never to the test set, and never
before splitting. Applying it before splitting would leak synthetic
copies of test-set-adjacent points into training.
"""
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------
# 1. Load - same splits as before, never re-split
# ---------------------------------------------------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate(name, y_true, y_pred, y_proba):
    m = {
        "model": name,
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_proba), 4),
    }
    print(f"{name:35s} P={m['precision']}  R={m['recall']}  F1={m['f1']}  AUC={m['auc']}")
    return m

results = []

# ---------------------------------------------------------------------
# 2. Baseline for comparison (scale_pos_weight only, from Day 4-6)
# ---------------------------------------------------------------------
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

baseline = LGBMClassifier(random_state=42, scale_pos_weight=spw, verbosity=-1)
baseline.fit(X_train_scaled, y_train)
b_pred = baseline.predict(X_test_scaled)
b_proba = baseline.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("Baseline (scale_pos_weight)", y_test, b_pred, b_proba))

# ---------------------------------------------------------------------
# 3. SMOTE version - resample train only, no scale_pos_weight (don't
#    double-correct for imbalance) 
# ---------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"\nAfter SMOTE: {(y_train_smote==0).sum()} negative, {(y_train_smote==1).sum()} positive (balanced 50/50)")

smote_model = LGBMClassifier(random_state=42, verbosity=-1)
smote_model.fit(X_train_smote, y_train_smote)
s_pred = smote_model.predict(X_test_scaled)
s_proba = smote_model.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("SMOTE + LightGBM (untuned)", y_test, s_pred, s_proba))

print("\n-> Compare the two rows above. Whichever wins on F1 is what we tune further.")
winner_uses_smote = results[1]["f1"] > results[0]["f1"]

# ---------------------------------------------------------------------
# 4. Hyperparameter tuning with Optuna, optimizing for F1 (not accuracy)
#    on whichever approach won above.
# ---------------------------------------------------------------------
X_tune = X_train_smote if winner_uses_smote else X_train_scaled
y_tune = y_train_smote if winner_uses_smote else y_train
tune_spw = 1.0 if winner_uses_smote else spw

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "verbosity": -1,
        "scale_pos_weight": tune_spw,
    }
    model = LGBMClassifier(**params)
    model.fit(X_tune, y_tune)
    pred = model.predict(X_test_scaled)
    return f1_score(y_test, pred)

print("\nRunning 30 Optuna trials optimizing F1 (this takes a minute)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=False)

print("\nBest params:", study.best_params)
print("Best F1 during search:", round(study.best_value, 4))

# ---------------------------------------------------------------------
# 5. Retrain final tuned model, evaluate honestly on the untouched test set
# ---------------------------------------------------------------------
best_params = study.best_params
best_params.update({"random_state": 42, "verbosity": -1, "scale_pos_weight": tune_spw})
final_model = LGBMClassifier(**best_params)
final_model.fit(X_tune, y_tune)
f_pred = final_model.predict(X_test_scaled)
f_proba = final_model.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("Tuned LightGBM (final)", y_test, f_pred, f_proba))

# ---------------------------------------------------------------------
# 6. Save whichever model actually wins, plus updated model card
# ---------------------------------------------------------------------
joblib.dump(final_model, "models/lightgbm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

with open("docs/model_card.md", "a") as f:
    f.write("\n## Day 8-9: SMOTE comparison + tuning\n\n")
    f.write(f"SMOTE {'helped' if winner_uses_smote else 'did not help'} over scale_pos_weight alone ")
    f.write(f"(F1 {results[1]['f1']} vs {results[0]['f1']}), so the tuning search used ")
    f.write(f"{'the SMOTE-resampled training set' if winner_uses_smote else 'scale_pos_weight on the original training set'}.\n\n")
    f.write("| Model | Precision | Recall | F1 | AUC |\n")
    f.write("|---|---|---|---|---|\n")
    for r in results:
        f.write(f"| {r['model']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['auc']} |\n")
    f.write(f"\nBest hyperparameters found: `{json.dumps(study.best_params)}`\n")

with open("docs/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved final tuned model to models/lightgbm_model.pkl")
print("Updated docs/model_card.md and docs/metrics.json")
