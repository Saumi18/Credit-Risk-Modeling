# Model Card - Credit Risk Baseline Models

Dataset: UCI Default of Credit Card Clients (30,000 rows, 22.1% positive class)

Split: 80/20 stratified, random_state=42, scaler fit on train only (no leakage)

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.7445 | 0.4429 | 0.6021 | 0.5104 | 0.7449 |
| LightGBM | 0.7567 | 0.4627 | 0.6209 | 0.5302 | 0.7778 |

## Notes

- Accuracy alone is misleading here: a model predicting 'no default' for
  every row scores ~78% accuracy while being useless. Precision/recall/F1/AUC
  are the metrics that actually matter for this imbalanced problem.
- Both models use class-imbalance handling (class_weight/scale_pos_weight)
  rather than plain defaults, which would otherwise barely predict the
  minority (default) class at all.
- These are BASELINE, untuned models. Next steps: SMOTE comparison,
  hyperparameter tuning, threshold tuning on the precision-recall curve.
