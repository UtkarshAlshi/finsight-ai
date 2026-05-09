FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen

COPY src/ ./src/

# Pre-download models at build time so container starts cold
RUN uv run python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert', device=-1)"

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/app/models

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uv", "run", "uvicorn", "src.finsight.ml.sentiment_service:app", \
     "--host", "0.0.0.0", "--port", "8001"]
