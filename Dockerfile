# Sentinel scoring API
#   python scripts/export_model.py            # once: download fraud_model@champion
#   docker build -t sentinel-api .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" sentinel-api
#   open http://localhost:8000/docs
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install ".[ml,serve]"

RUN useradd --create-home sentinel
USER sentinel

ENV SENTINEL_MODEL_PATH=/app/models/champion
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "sentinel.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
