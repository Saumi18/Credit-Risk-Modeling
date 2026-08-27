"""
src/model.py

Loads the trained LightGBM model. No training logic here - training
happens in notebooks/, this module only loads and serves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_model(config: dict[str, Any]):
    path = config["model_paths"]["lightgbm"]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run notebooks/03_baseline_models.py "
            f"and notebooks/04_smote_tuning.py first to produce it."
        )
    return joblib.load(path)
