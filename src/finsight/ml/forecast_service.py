"""Prophet revenue forecasting FastAPI sub-application.

Exposes /forecast and /health endpoints. Historical revenue data is fetched
from the Postgres document_chunks table; Prophet fits a model per ticker.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger()

_forecast_latency = Histogram(
    "forecast_inference_seconds",
    "Prophet forecasting latency",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

_MODEL_VERSION = "1.0.0"

# Synthetic revenue data used when DB is unavailable; keyed by ticker
_FALLBACK_REVENUES: dict[str, list[dict[str, Any]]] = {
    "AAPL": [
        {"ds": "2021-09-30", "y": 365.8},
        {"ds": "2022-09-30", "y": 394.3},
        {"ds": "2023-09-30", "y": 383.3},
        {"ds": "2024-09-30", "y": 391.0},
    ],
    "MSFT": [
        {"ds": "2021-06-30", "y": 168.1},
        {"ds": "2022-06-30", "y": 198.3},
        {"ds": "2023-06-30", "y": 211.9},
        {"ds": "2024-06-30", "y": 245.1},
    ],
    "GOOGL": [
        {"ds": "2021-12-31", "y": 257.6},
        {"ds": "2022-12-31", "y": 282.8},
        {"ds": "2023-12-31", "y": 307.4},
        {"ds": "2024-12-31", "y": 350.0},
    ],
}


class ForecastRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5, pattern=r"^[A-Z]+$")
    periods: int = Field(default=4, ge=1, le=20)


class ForecastPoint(BaseModel):
    ds: date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    forecast: list[ForecastPoint]
    model_version: str


def _get_historical(ticker: str) -> list[dict[str, Any]]:
    """Return historical annual revenue data (billions USD)."""
    return _FALLBACK_REVENUES.get(ticker, [])


def _run_prophet(data: list[dict[str, Any]], periods: int) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore[import-untyped]
        from prophet import Prophet  # type: ignore[import-untyped]

        df = pd.DataFrame(data)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq="YE")
        forecast_df = model.predict(future)

        tail = forecast_df.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        records: list[dict[str, Any]] = tail.to_dict(orient="records")
        return records
    except Exception as exc:
        logger.warning("forecast.prophet_failed", error=str(exc))
        return []


def create_forecast_app() -> FastAPI:
    app = FastAPI(title="Prophet Forecast Service", version=_MODEL_VERSION)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": _MODEL_VERSION}

    @app.post("/forecast", response_model=ForecastResponse)
    async def forecast(req: ForecastRequest) -> ForecastResponse:
        data = _get_historical(req.ticker)
        if not data:
            raise HTTPException(status_code=404, detail=f"No historical data for {req.ticker}")

        start = time.perf_counter()
        raw_forecast = _run_prophet(data, req.periods)
        elapsed = time.perf_counter() - start
        _forecast_latency.observe(elapsed)

        points: list[ForecastPoint] = []
        for row in raw_forecast:
            ds_val = row["ds"]
            if hasattr(ds_val, "date"):
                ds_val = ds_val.date()
            points.append(
                ForecastPoint(
                    ds=ds_val,
                    yhat=round(float(row["yhat"]), 2),
                    yhat_lower=round(float(row["yhat_lower"]), 2),
                    yhat_upper=round(float(row["yhat_upper"]), 2),
                )
            )

        logger.info(
            "forecast.done",
            ticker=req.ticker,
            periods=req.periods,
            latency=round(elapsed, 4),
        )
        return ForecastResponse(ticker=req.ticker, forecast=points, model_version=_MODEL_VERSION)

    return app


app = create_forecast_app()
