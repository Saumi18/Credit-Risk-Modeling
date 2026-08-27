"""
api/main.py

FastAPI backend wrapping src/ - the exact same preprocessing, feature
engineering, and model used in the notebooks, now served over HTTP.

Run locally with:
    uvicorn api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs
(FastAPI generates this automatically from the Pydantic schemas).
"""
import io
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from src.preprocessing import load_config, load_scaler, prepare_features, SchemaError
from src.model import load_model
from src.inference import predict

from api.schemas import (
    CreditApplication, PredictionResponse, BatchPredictionResponse,
    ModelInfo, HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Decision API",
    description="Predicts probability of credit default using a tuned LightGBM model.",
    version="1.0.0",
)

# ---------------------------------------------------------------------
# Load config, scaler, and model ONCE at startup - not per-request.
# Reloading a model on every request would make the API unusably slow.
# ---------------------------------------------------------------------
config = load_config()
scaler = load_scaler(config["model_paths"]["scaler"])
model = load_model(config)
threshold = config["decision_threshold"]

logger.info(f"Model loaded. Decision threshold = {threshold}")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return {
        "model_name": "LightGBM (tuned, Optuna)",
        "decision_threshold": threshold,
        "n_features": len(config["model_feature_columns"]),
        "feature_columns": config["model_feature_columns"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(application: CreditApplication):
    df = pd.DataFrame([application.model_dump()])
    try:
        X = prepare_features(df, config, scaler)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = predict(model, X, threshold)
    return {
        "prediction": result.predictions[0],
        "probability": result.probabilities[0],
        "threshold_used": result.threshold_used,
    }


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse file as CSV.")

    try:
        X = prepare_features(df, config, scaler)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = predict(model, X, threshold)
    return {
        "predictions": result.predictions,
        "probabilities": result.probabilities,
        "threshold_used": result.threshold_used,
        "n_rows": result.n_rows,
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
