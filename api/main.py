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
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.preprocessing import load_config, load_scaler, prepare_features, SchemaError
from src.model import load_model
from src.inference import predict

from api.schemas import (
    CreditApplication, PredictionResponse, BatchPredictionResponse,
    ModelInfo, HealthResponse,
)
from database.connection import get_db, init_db
from database.models import Prediction, ModelVersion
from database.cache import get_cached_prediction, set_cached_prediction
from workers.queue_client import enqueue_batch_job, get_job_status

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

MODEL_VERSION_ID = "lightgbm_tuned_v1"  # bump this string if you retrain/redeploy a new model


@app.on_event("startup")
def on_startup():
    """Creates DB tables and registers this model version, if not already present.
    Runs once when the API process starts, not per-request."""
    try:
        init_db()
        from database.connection import SessionLocal
        db = SessionLocal()
        existing = db.get(ModelVersion, MODEL_VERSION_ID)
        if not existing:
            db.add(ModelVersion(id=MODEL_VERSION_ID, name="LightGBM (tuned, Optuna)", threshold=threshold))
            db.commit()
        db.close()
        logger.info("Database ready.")
    except Exception as e:
        logger.warning(f"Database unavailable at startup ({e}). "
                        f"API will still serve predictions, but won't persist them.")


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
def predict_single(application: CreditApplication, db: Session = Depends(get_db)):
    payload = application.model_dump()

    cached = get_cached_prediction(payload)
    if cached:
        return cached

    df = pd.DataFrame([payload])
    try:
        X = prepare_features(df, config, scaler)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = predict(model, X, threshold)
    response = {
        "prediction": result.predictions[0],
        "probability": result.probabilities[0],
        "threshold_used": result.threshold_used,
    }

    set_cached_prediction(payload, response)

    try:
        db.add(Prediction(
            model_version_id=MODEL_VERSION_ID,
            input_data=payload,
            prediction=response["prediction"],
            probability=response["probability"],
        ))
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist prediction: {e}")
        db.rollback()

    return response


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Enqueues the batch job and returns immediately with a job_id.
    This does NOT block waiting for scoring to finish - for a 10,000-row
    CSV, this returns in milliseconds instead of however long scoring takes.
    Check progress with GET /batch-predict/{job_id}."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()

    # Quick shape check before even enqueueing, so obviously bad files
    # fail fast instead of taking a worker slot for no reason.
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse file as CSV.")

    try:
        job_id = enqueue_batch_job(contents)
    except Exception as e:
        logger.exception("Failed to enqueue batch job")
        raise HTTPException(status_code=503, detail=f"Could not enqueue job: {e}")

    return {"job_id": job_id, "status": "queued", "n_rows_submitted": len(df)}


@app.get("/batch-predict/{job_id}")
def batch_predict_status(job_id: str):
    """Poll this to check on a submitted batch job."""
    status = get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found.")
    return status


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    record = db.get(Prediction, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {
        "id": record.id,
        "input_data": record.input_data,
        "prediction": record.prediction,
        "probability": record.probability,
        "created_at": record.created_at.isoformat(),
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
