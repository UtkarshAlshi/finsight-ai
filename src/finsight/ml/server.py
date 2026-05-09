"""Unified ML service server — mounts sentiment, forecast, anomaly sub-apps."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from finsight.ml.anomaly_service import create_anomaly_app
from finsight.ml.forecast_service import create_forecast_app
from finsight.ml.sentiment_service import create_sentiment_app


def create_ml_app() -> FastAPI:
    app = FastAPI(title="FinSight ML Services", version="1.0.0")

    app.mount("/sentiment", create_sentiment_app())
    app.mount("/forecast", create_forecast_app())
    app.mount("/anomaly", create_anomaly_app())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


ml_app = create_ml_app()

if __name__ == "__main__":
    uvicorn.run("finsight.ml.server:ml_app", host="0.0.0.0", port=8001, reload=False)
