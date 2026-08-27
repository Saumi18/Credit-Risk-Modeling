"""
api/schemas.py

Request/response models matching config/features.yaml's raw_input_columns
exactly. Pydantic validates types automatically - this replaces the kind
of manual "did they send the right columns" checks with something FastAPI
enforces before your code even runs.
"""
from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    """One applicant's raw data - the 23 original UCI columns."""
    LIMIT_BAL: float = Field(..., description="Credit limit (NT dollar)")
    SEX: int = Field(..., ge=1, le=2, description="1=male, 2=female")
    EDUCATION: int = Field(..., ge=0, le=6)
    MARRIAGE: int = Field(..., ge=0, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
                "PAY_1": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
                "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
                "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
                "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
                "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no default, 1 = predicted default")
    probability: float = Field(..., description="Predicted probability of default")
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    threshold_used: float
    n_rows: int


class ModelInfo(BaseModel):
    model_name: str
    decision_threshold: float
    n_features: int
    feature_columns: list[str]


class HealthResponse(BaseModel):
    status: str
