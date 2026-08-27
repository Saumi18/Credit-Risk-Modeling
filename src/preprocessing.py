"""
src/preprocessing.py

Validates incoming data against the real trained schema (config/features.yaml)
and applies the same scaler used at training time. Fails loudly with a
specific message instead of letting sklearn throw a cryptic shape error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from .features import engineer_features, clean_categorical_codes

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "features.yaml"


class SchemaError(ValueError):
    """Raised when incoming data doesn't match the expected raw schema."""


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_scaler(path: str | Path):
    return joblib.load(path)


def validate_raw_schema(df: pd.DataFrame, config: dict[str, Any]) -> None:
    required = set(config["raw_input_columns"])
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(
            f"Missing required columns: {sorted(missing)}. "
            f"Expected raw input columns: {sorted(required)}."
        )


def prepare_features(df: pd.DataFrame, config: dict[str, Any], scaler) -> pd.DataFrame:
    """Full pipeline: validate -> clean -> engineer -> reorder -> scale.
    This is the single source of truth used by both the API and any
    batch scoring job, so training and serving never drift apart."""
    validate_raw_schema(df, config)

    cleaned = clean_categorical_codes(df)
    engineered = engineer_features(cleaned)

    model_cols = config["model_feature_columns"]
    ordered = engineered[model_cols].copy()

    scaled_array = scaler.transform(ordered)
    return pd.DataFrame(scaled_array, columns=model_cols, index=ordered.index)
