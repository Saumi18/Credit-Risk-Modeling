"""
tests/test_database.py

Uses SQLite in-memory instead of real Postgres, and doesn't require Redis
to be running - this is the standard pattern for fast, isolated tests
that don't depend on external services being up.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, ModelVersion, Prediction


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_model_version(db_session):
    mv = ModelVersion(id="test_v1", name="LightGBM test", threshold=0.5)
    db_session.add(mv)
    db_session.commit()

    fetched = db_session.get(ModelVersion, "test_v1")
    assert fetched.name == "LightGBM test"
    assert fetched.threshold == 0.5


def test_create_prediction_linked_to_model_version(db_session):
    mv = ModelVersion(id="test_v1", name="LightGBM test", threshold=0.5)
    db_session.add(mv)
    db_session.commit()

    pred = Prediction(
        model_version_id="test_v1",
        input_data={"LIMIT_BAL": 20000, "AGE": 24},
        prediction=1,
        probability=0.88,
    )
    db_session.add(pred)
    db_session.commit()

    fetched = db_session.get(Prediction, pred.id)
    assert fetched.prediction == 1
    assert fetched.probability == 0.88
    assert fetched.input_data["AGE"] == 24
    assert fetched.model_version.name == "LightGBM test"


def test_cache_key_is_deterministic():
    from database.cache import _cache_key
    payload_a = {"AGE": 24, "LIMIT_BAL": 20000}
    payload_b = {"LIMIT_BAL": 20000, "AGE": 24}  # same content, different key order
    assert _cache_key(payload_a) == _cache_key(payload_b)


def test_cache_key_differs_for_different_payloads():
    from database.cache import _cache_key
    assert _cache_key({"AGE": 24}) != _cache_key({"AGE": 25})
