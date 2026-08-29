# Dockerfile
#
# One image, used by BOTH the api and worker services in docker-compose.yml
# (they run identical code, just different entrypoint commands). This
# avoids maintaining two near-duplicate Dockerfiles that could drift apart.

FROM python:3.11-slim

# lightgbm needs libgomp1 at runtime; psycopg2-binary avoids needing
# postgres dev headers, but we still need basic build tools for a few
# packages that compile from source on some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first and install - this layer only rebuilds when
# requirements.txt changes, not on every code change, which makes
# rebuilds during development much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY api/ ./api/
COPY src/ ./src/
COPY database/ ./database/
COPY workers/ ./workers/
COPY config/ ./config/
COPY models/ ./models/
COPY app.py .

EXPOSE 8000

# Default command runs the API. The worker service in docker-compose.yml
# overrides this with its own command.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
