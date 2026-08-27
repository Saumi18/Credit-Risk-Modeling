"""
tests/test_pipeline.py

End-to-end test: takes raw-shaped data (like a real API request would),
runs it through the full src/ pipeline, and confirms it produces valid
predictions using the ACTUAL trained model and scaler - not a mock.
"""
import pandas as pd
import pytest

from src.preprocessing import load_config, load_scaler, prepare_features, validate_raw_schema, SchemaError
from src.model import load_model
from src.inference import predict


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def scaler(config):
    return load_scaler(config["model_paths"]["scaler"])


@pytest.fixture(scope="module")
def model(config):
    return load_model(config)


@pytest.fixture
def sample_raw_row():
    """One realistic row shaped exactly like real UCI input - no engineered
    columns, mimicking what an API client would actually send."""
    return pd.DataFrame([{
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_1": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0,
        "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0,
        "PAY_AMT5": 0, "PAY_AMT6": 0,
    }])


def test_validate_raw_schema_passes_with_real_columns(sample_raw_row, config):
    validate_raw_schema(sample_raw_row, config)  # should not raise


def test_validate_raw_schema_fails_with_missing_column(config):
    bad_df = pd.DataFrame([{"LIMIT_BAL": 20000}])
    with pytest.raises(SchemaError):
        validate_raw_schema(bad_df, config)


def test_full_pipeline_produces_valid_prediction(sample_raw_row, config, scaler, model):
    X = prepare_features(sample_raw_row, config, scaler)
    assert X.shape == (1, len(config["model_feature_columns"]))

    result = predict(model, X, config["decision_threshold"])
    assert len(result.predictions) == 1
    assert result.predictions[0] in (0, 1)
    assert 0.0 <= result.probabilities[0] <= 1.0
    assert result.threshold_used == config["decision_threshold"]


def test_pipeline_handles_multiple_rows(sample_raw_row, config, scaler, model):
    batch = pd.concat([sample_raw_row] * 5, ignore_index=True)
    X = prepare_features(batch, config, scaler)
    result = predict(model, X, config["decision_threshold"])
    assert result.n_rows == 5
    assert len(result.predictions) == 5
