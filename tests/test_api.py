"""
tests/test_api.py

Tests the real FastAPI app in-process using TestClient - no mocking of
the model, scaler, or preprocessing. This loads your actual .pkl files.
"""
import io
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_APPLICATION = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_1": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] == 30
    assert "months_late" in body["feature_columns"]


def test_predict_valid_application():
    resp = client.post("/predict", json=SAMPLE_APPLICATION)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_missing_field_returns_422():
    bad_application = SAMPLE_APPLICATION.copy()
    del bad_application["LIMIT_BAL"]
    resp = client.post("/predict", json=bad_application)
    assert resp.status_code == 422  # FastAPI/Pydantic validation error


def test_batch_predict_valid_csv():
    df = pd.DataFrame([SAMPLE_APPLICATION, SAMPLE_APPLICATION])
    csv_bytes = df.to_csv(index=False).encode()
    resp = client.post(
        "/batch-predict",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 2
    assert len(body["predictions"]) == 2


def test_batch_predict_rejects_non_csv():
    resp = client.post(
        "/batch-predict",
        files={"file": ("test.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert resp.status_code == 400


def test_batch_predict_missing_columns_returns_400():
    df = pd.DataFrame([{"LIMIT_BAL": 20000}])  # missing everything else
    csv_bytes = df.to_csv(index=False).encode()
    resp = client.post(
        "/batch-predict",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 400
