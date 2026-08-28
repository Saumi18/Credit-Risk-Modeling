"""
database/models.py

Three tables, matching the plan from the very first conversation:
  - customers: who applied (optional link if you want to track repeat applicants)
  - predictions: every prediction the API ever made, with input + output
  - model_versions: which model version produced which prediction (so if you
    retrain later, you can tell old predictions from new ones)
"""
import datetime
import uuid

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)          # e.g. "lightgbm_tuned"
    threshold = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)

    input_data = Column(JSON, nullable=False)       # the raw 23-column request, for auditability
    prediction = Column(Integer, nullable=False)    # 0 or 1
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="predictions")
