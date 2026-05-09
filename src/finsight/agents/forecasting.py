"""Forecasting agent: calls Prophet ML service for revenue/price forecasts."""

from __future__ import annotations

from typing import Any

import aiohttp
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from finsight.agents.state import GraphState
from finsight.config import Settings
from finsight.errors import MLServiceError

logger = structlog.get_logger()


async def forecasting_node(state: GraphState, settings: Settings) -> dict[str, Any]:
    """LangGraph node: request revenue forecast from Prophet ML service."""
    tickers = _extract_tickers_from_question(state.question)
    if not tickers:
        return {"ml_outputs": {**state.ml_outputs, "forecast": None}}

    forecasts: dict[str, Any] = {}
    for ticker in tickers[:3]:  # limit concurrent forecasts
        forecast = await _call_forecast_service(ticker)
        if forecast:
            forecasts[ticker] = forecast

    logger.info("forecasting.done", tickers=list(forecasts.keys()))
    return {"ml_outputs": {**state.ml_outputs, "forecast": forecasts if forecasts else None}}


async def _call_forecast_service(ticker: str) -> dict[str, Any] | None:
    ml_url = "http://localhost:8001/forecast"
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(2),
            wait=wait_exponential(min=1, max=5),
            reraise=True,
        ):
            with attempt:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        ml_url,
                        json={"ticker": ticker, "periods": 4},
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status != 200:
                            raise MLServiceError(f"Forecast service error: {resp.status}")
                        return await resp.json()  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("forecasting.service_unavailable", ticker=ticker, error=str(exc))
        return None
    return None


def _extract_tickers_from_question(question: str) -> list[str]:
    import re

    known = {"META", "GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "JPM", "V", "WMT", "GOOG"}
    found = re.findall(r"\b([A-Z]{1,5})\b", question)
    return [t for t in found if t in known]
