"""Isolation Forest anomaly detection FastAPI sub-application.

Detects anomalous financial metrics (e.g. revenue spike, unusual ratios).
Exposes /detect and /health endpoints. Model is fit per-request on the
provided time-series features; no persistent model state is required.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import FastAPI
from prometheus_client import Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger()

_anomaly_latency = Histogram(
    "anomaly_inference_seconds",
    "Isolation Forest inference latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

_MODEL_VERSION = "1.0.0"
_CONTAMINATION = 0.1


class AnomalyRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5)
    features: list[list[float]] = Field(
        ...,
        description="Each inner list is one observation's feature vector",
        min_length=5,
    )
    feature_names: list[str] = Field(default_factory=list)


class AnomalyPoint(BaseModel):
    index: int
    score: float
    is_anomaly: bool


class AnomalyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    results: list[AnomalyPoint]
    anomaly_count: int
    model_version: str


def _run_isolation_forest(features: list[list[float]]) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

        X = np.array(features, dtype=np.float64)
        clf = IsolationForest(contamination=_CONTAMINATION, random_state=42, n_estimators=100)
        clf.fit(X)

        scores = clf.decision_function(X)
        preds = clf.predict(X)

        return [
            {"index": i, "score": float(scores[i]), "is_anomaly": bool(preds[i] == -1)}
            for i in range(len(features))
        ]
    except Exception as exc:
        logger.warning("anomaly.sklearn_failed", error=str(exc))
        return [{"index": i, "score": 0.0, "is_anomaly": False} for i in range(len(features))]


def create_anomaly_app() -> FastAPI:
    app = FastAPI(title="Anomaly Detection Service", version=_MODEL_VERSION)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": _MODEL_VERSION}

    @app.post("/detect", response_model=AnomalyResponse)
    async def detect(req: AnomalyRequest) -> AnomalyResponse:
        start = time.perf_counter()
        raw = _run_isolation_forest(req.features)
        elapsed = time.perf_counter() - start
        _anomaly_latency.observe(elapsed)

        results = [AnomalyPoint(**r) for r in raw]
        anomaly_count = sum(1 for r in results if r.is_anomaly)

        logger.info(
            "anomaly.detect",
            ticker=req.ticker,
            observations=len(req.features),
            anomalies=anomaly_count,
            latency=round(elapsed, 4),
        )
        return AnomalyResponse(
            ticker=req.ticker,
            results=results,
            anomaly_count=anomaly_count,
            model_version=_MODEL_VERSION,
        )

    return app


app = create_anomaly_app()
