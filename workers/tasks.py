"""
workers/tasks.py

The function that actually runs INSIDE a worker process, not the API
process. This reuses the exact same src/ pipeline as /predict, so a
batch job and a single prediction never compute things differently.

Results are written to Postgres (predictions table, same as single
predictions) so /predictions/{id} works identically for batch results too.
"""
import io
import logging

import pandas as pd

from src.preprocessing import load_config, load_scaler, prepare_features, SchemaError
from src.model import load_model
from src.inference import predict
from database.connection import SessionLocal
from database.models import Prediction

logger = logging.getLogger(__name__)

# Loaded once per WORKER PROCESS (not per job) - same reasoning as the API:
# reloading a model for every row of every batch would be extremely slow.
_config = load_config()
_scaler = load_scaler(_config["model_paths"]["scaler"])
_model = load_model(_config)
_threshold = _config["decision_threshold"]

MODEL_VERSION_ID = "lightgbm_tuned_v1"


def process_batch(csv_bytes: bytes) -> dict:
    """Runs in the worker process. Returns a JSON-serializable summary;
    RQ stores this as the job's result automatically."""
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        return {"status": "failed", "error": f"Could not parse CSV: {e}"}

    try:
        X = prepare_features(df, _config, _scaler)
    except SchemaError as e:
        return {"status": "failed", "error": str(e)}

    result = predict(_model, X, _threshold)

    # Persist each row as its own Prediction record, same table single
    # predictions use, so results are queryable the same way either path.
    prediction_ids = []
    db = SessionLocal()
    try:
        for i in range(result.n_rows):
            record = Prediction(
                model_version_id=MODEL_VERSION_ID,
                input_data=df.iloc[i].to_dict(),
                prediction=result.predictions[i],
                probability=result.probabilities[i],
            )
            db.add(record)
            db.flush()  # get the generated id without committing yet
            prediction_ids.append(record.id)
        db.commit()
    except Exception as e:
        logger.exception("Failed to persist batch predictions")
        db.rollback()
        return {"status": "failed", "error": f"Database error: {e}"}
    finally:
        db.close()

    return {
        "status": "completed",
        "n_rows": result.n_rows,
        "prediction_ids": prediction_ids,
        "predictions": result.predictions,
        "probabilities": result.probabilities,
    }
