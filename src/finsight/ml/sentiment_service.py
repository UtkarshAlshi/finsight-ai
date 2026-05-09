"""FinBERT sentiment FastAPI sub-application.

Exposes /predict and /health endpoints. Model is loaded once at startup and
cached in module state. Prometheus histograms track inference latency.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from prometheus_client import Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger()

_inference_latency = Histogram(
    "sentiment_inference_seconds",
    "FinBERT inference latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

_MODEL_NAME = "ProsusAI/finbert"
_MODEL_VERSION = "1.0.0"

_pipeline: Any = None


def _load_model() -> Any:
    global _pipeline  # noqa: PLW0603
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline  # type: ignore[import-untyped]

        _pipeline = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            top_k=None,
            device=-1,
        )
        logger.info("sentiment_model.loaded", model=_MODEL_NAME, version=_MODEL_VERSION)
    except Exception as exc:
        logger.warning("sentiment_model.load_failed", error=str(exc))
        _pipeline = None
    return _pipeline


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)


class SentimentScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[SentimentScore]
    model_version: str


def create_sentiment_app() -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        _load_model()
        yield

    app = FastAPI(title="FinBERT Sentiment Service", version=_MODEL_VERSION, lifespan=_lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": _MODEL_NAME, "version": _MODEL_VERSION}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        model = _load_model()
        results: list[SentimentScore] = []

        start = time.perf_counter()
        if model is None:
            # Graceful degradation: return neutral scores
            for _ in req.texts:
                results.append(SentimentScore(label="neutral", score=1.0))
        else:
            raw = model(req.texts, truncation=True, max_length=512)
            for text_scores in raw:
                # Each text_scores is a list of {label, score}; pick top label
                if isinstance(text_scores, list) and text_scores:
                    best = max(text_scores, key=lambda x: x["score"])
                    results.append(SentimentScore(label=best["label"].lower(), score=best["score"]))
                else:
                    results.append(SentimentScore(label="neutral", score=1.0))

        elapsed = time.perf_counter() - start
        _inference_latency.observe(elapsed)
        logger.info("sentiment.predict", texts=len(req.texts), latency=round(elapsed, 4))

        return PredictResponse(predictions=results, model_version=_MODEL_VERSION)

    return app


app = create_sentiment_app()
