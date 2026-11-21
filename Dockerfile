# syntax=docker/dockerfile:1
# Reproducible serving image for the registered MiBici HGB champion.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MIBICI_MODEL_URI=/app/models/serving/champion \
    MIBICI_SNAPSHOT_PATH=/app/models/serving/history_snapshot.parquet \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Only dependency files are copied before installation, so source changes reuse this layer.
COPY requirements-serving.txt /app/requirements-serving.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-serving.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser
COPY --chown=appuser:appuser mibici/ /app/mibici/
COPY --chown=appuser:appuser models/serving/champion/ /app/models/serving/champion/
COPY --chown=appuser:appuser models/serving/history_snapshot.parquet /app/models/serving/history_snapshot.parquet
COPY --chown=appuser:appuser models/serving/snapshot.json /app/models/serving/snapshot.json

# Detect missing artifacts during build instead of discovering them at runtime.
RUN python -c "from pathlib import Path; import json; p=Path('/app/models/serving'); assert (p/'champion'/'MLmodel').is_file(); assert (p/'champion'/'model.pkl').is_file(); assert (p/'history_snapshot.parquet').is_file(); assert json.loads((p/'snapshot.json').read_text())['digest'] == '66bc9009'"

USER appuser
EXPOSE 8000
# Feature construction at startup is intentionally included in the health period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "mibici.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
