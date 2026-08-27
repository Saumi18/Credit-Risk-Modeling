"""
src/inference.py

Prediction logic using the F1-optimal threshold from Day 10
(config/features.yaml: decision_threshold), not the arbitrary 0.5 default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PredictionResult:
    predictions: list[int]
    probabilities: list[float]
    threshold_used: float
    n_rows: int


def predict(model: Any, X: pd.DataFrame, threshold: float) -> PredictionResult:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > threshold).astype(int)

    return PredictionResult(
        predictions=pred.tolist(),
        probabilities=proba.tolist(),
        threshold_used=threshold,
        n_rows=len(X),
    )
