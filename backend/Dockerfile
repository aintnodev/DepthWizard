# DEPTHWIZARD backend — Python 3.12, FastAPI, CPU-friendly
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by rasterio/shapely wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/
RUN pip install --upgrade pip && pip install -r backend/requirements.txt

# Copy application code
COPY backend/app ./app

# Jobs artefacts + HF weight cache
RUN mkdir -p /app/data/jobs

EXPOSE 8000

# CPU inference by default; the API reports the device via /api/health so the
# UI can clearly indicate "CPU inference" when no GPU is present.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
