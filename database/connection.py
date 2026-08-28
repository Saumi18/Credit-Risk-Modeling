"""
database/connection.py

Reads connection info from environment variables (never hardcode
credentials in source), with sensible local-dev defaults. This is the
standard pattern - it's what lets the SAME code run against your local
Postgres and against a production database later, just by changing
environment variables, not code.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/credit_risk",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency - yields a session, always closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't exist. Called once at startup."""
    from database.models import Base
    Base.metadata.create_all(bind=engine)
