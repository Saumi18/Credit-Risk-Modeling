# Credit Risk Decision System

A full-stack, production-style ML system that predicts the probability of
credit card default, built end-to-end: data cleaning and feature engineering,
model tuning and explainability, a FastAPI backend with async batch
processing, PostgreSQL + Redis, full Docker containerization, and a live
deployed web app.

**Live demo:** https://credit-risk-modeling.vercel.app
**API docs (Swagger):** https://credit-risk-api-7n7s.onrender.com/docs

> Note: the backend runs on a free-tier host that sleeps after 15 minutes
> of inactivity. The first prediction after a period of inactivity may take
> 30-60 seconds while it wakes up - the UI shows a loading message during
> this time. Subsequent requests are instant.

---

## What it does

Given an applicant's credit history (credit limit, demographics, and the
last 6 months of payment behavior), the system predicts:
- **Probability of default** (0-100%)
- **Risk classification** (LOW / HIGH), using an F1-optimized decision
  threshold rather than an arbitrary 50% cutoff

## Screenshots

*(Add 1-2 screenshots of the live UI here before sharing this repo -
a screenshot of the form and a screenshot of a completed prediction
with the risk gauge go a long way for anyone skimming the repo.)*

---

## Architecture

```
                        ┌─────────────────┐
                        │  Frontend (UI)   │  Vercel
                        │  HTML/JS/Tailwind│
                        └────────┬─────────┘
                                 │ HTTPS (CORS-enabled)
                                 ▼
                        ┌─────────────────┐
                        │   FastAPI        │  Render
                        │   /predict       │
                        │   /batch-predict │
                        │   /model/info    │
                        └───┬─────────┬────┘
                            │         │
                   ┌────────▼──┐   ┌──▼─────────┐
                   │ PostgreSQL │   │   Redis    │
                   │ predictions│   │ cache +    │
                   │ model_vers.│   │ job queue  │
                   └────────────┘   └─────┬──────┘
                                          │
                                   ┌──────▼──────┐
                                   │  RQ Worker   │
                                   │ batch scoring│
                                   └──────────────┘
```

Single predictions are served synchronously with Redis-backed caching.
Batch predictions (CSV upload) are enqueued and processed asynchronously
by a separate worker, so large files don't block the API.

## Tech stack

| Layer | Technology |
|---|---|
| Model | LightGBM (tuned via Optuna), scikit-learn |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Cache / Queue | Redis, RQ |
| Containerization | Docker, Docker Compose |
| Frontend | HTML, Tailwind CSS, vanilla JS |
| Deployment | Render (backend), Vercel (frontend) |
| Testing | pytest, FastAPI TestClient |

---

## Model performance

Trained on the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(30,000 rows, ~22% positive class - imbalanced).

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.443 | 0.602 | 0.510 | 0.745 |
| LightGBM (baseline) | 0.463 | 0.621 | 0.530 | 0.778 |
| LightGBM (tuned, 0.5 threshold) | 0.484 | 0.617 | 0.542 | 0.778 |
| **LightGBM (tuned, final threshold=0.5109)** | - | - | **0.545** | **0.778** |

Decision threshold: **0.5109** (F1-optimized via precision-recall curve
analysis, not the default 0.5) — improves F1 from 0.542 to 0.545 over the
naive 0.5 cutoff.

A recall-favoring alternative threshold (0.345) was also evaluated,
achieving 77.3% recall at a 0.35 precision floor - useful context if a
real cost analysis later determines that missing a defaulter (false
negative) is costlier than a false alarm.

Full methodology, including why SMOTE was tested and rejected, and what
the top SHAP features were, is in [`docs/model_card.md`](docs/model_card.md).

**Honest context:** this is a well-known hard benchmark dataset - published
results on it typically land in the 0.75-0.82 AUC range, not higher. These
numbers reflect a properly validated, leakage-free pipeline, not an
inflated metric.

---

## Project structure

```
Credit-Risk-Modeling/
├── notebooks/          # EDA, cleaning, feature engineering, model training
├── src/                # Production preprocessing/model/inference (used by API)
├── api/                # FastAPI app: routes, schemas
├── database/           # SQLAlchemy models, connection, Redis cache
├── workers/            # Async batch job processing (RQ)
├── frontend/           # Static web UI
├── config/             # features.yaml - single source of truth for model schema
├── models/             # Trained model artifacts (.pkl)
├── data/               # Raw and processed datasets
├── docs/               # Model card, plots, metrics
├── tests/              # pytest suite (pipeline, API, database)
├── Dockerfile
├── docker-compose.yml  # Full local stack: api + worker + postgres + redis
├── render.yaml         # Render deployment blueprint
└── vercel.json         # Frontend deployment config
```

---

## Running locally

### Option A: Docker (recommended - runs the whole stack with one command)

```bash
docker compose up --build
```

This starts the API (port 8000), worker, PostgreSQL, and Redis together.
Then open `frontend/index.html` directly in your browser (it points at
`http://127.0.0.1:8000` by default).

### Option B: Native (without Docker)

```bash
pip install -r requirements.txt

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
python -m workers.worker
```

You'll also need PostgreSQL and Redis running locally (or via
`docker compose -f docker-compose.dev.yml up -d` for just those two).

### Running tests

```bash
python -m pytest tests/ -v
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata, feature list, threshold |
| POST | `/predict` | Single prediction |
| POST | `/batch-predict` | Enqueue a CSV for async batch scoring |
| GET | `/batch-predict/{job_id}` | Poll batch job status/results |
| GET | `/predictions/{id}` | Fetch a stored prediction by ID |

Full interactive docs (Swagger) available at `/docs` on any running instance.

---

## How this was built

This project was built incrementally in stages, each with real, tested
code before moving to the next:

1. Data cleaning + EDA (handling undocumented category codes, no leakage)
2. Feature engineering (utilization ratio, payment trends, delay counts)
3. Baseline models with honest imbalanced-classification metrics
4. SMOTE vs. class-weighting comparison (SMOTE was tested and rejected -
   it measurably hurt F1 here)
5. Hyperparameter tuning (Optuna) + threshold tuning (precision-recall
   curve analysis) + SHAP explainability
6. Extraction into production `src/` modules (shared by API and workers,
   zero drift between training and serving)
7. FastAPI backend with request validation and error handling
8. PostgreSQL persistence + Redis caching
9. Async batch processing (RQ) - including debugging a genuine
   `os.fork()` incompatibility on Windows, fixed with `SimpleWorker`
10. Full Docker containerization (api + worker + postgres + redis)
11. Live deployment (Render + Vercel), including a free-tier constraint
    (no standalone worker service) worked around by running the worker
    as a subprocess within the API service

## Known limitations

- Trained on 2005 Taiwanese credit card data - not calibrated for other
  populations or time periods. This is a portfolio/demonstration project,
  not a production lending system.
- Free-tier hosting means the backend sleeps after inactivity (see note
  at the top).
- The worker runs as a subprocess inside the API service in production
  (Render free tier doesn't support a separate worker service type);
  locally via Docker Compose it runs as a genuinely separate process,
  which is the architecturally correct setup.
