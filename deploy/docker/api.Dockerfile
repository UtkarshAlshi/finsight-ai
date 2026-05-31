FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen

COPY src/ ./src/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "src.finsight.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
