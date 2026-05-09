"""Sentiment agent: calls FinBERT ML service for earnings call sentiment."""

from __future__ import annotations

from typing import Any

import aiohttp
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from finsight.agents.state import GraphState
from finsight.config import Settings
from finsight.errors import MLServiceError

logger = structlog.get_logger()


async def sentiment_node(state: GraphState, settings: Settings) -> dict[str, Any]:
    """LangGraph node: run FinBERT sentiment on retrieved context snippets."""
    snippets = [
        chunk.get("text", "")[:512] for chunk in state.retrieved_context[:5] if chunk.get("text")
    ]
    if not snippets:
        return {"ml_outputs": {**state.ml_outputs, "sentiment": None}}

    scores = await _call_sentiment_service(snippets)
    aggregate = _aggregate_scores(scores)

    logger.info("sentiment.done", snippets=len(snippets), aggregate=aggregate)
    return {
        "ml_outputs": {**state.ml_outputs, "sentiment": {"scores": scores, "aggregate": aggregate}}
    }


async def _call_sentiment_service(texts: list[str]) -> list[dict[str, Any]]:
    ml_url = "http://localhost:8001/predict"
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
                        json={"texts": texts},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status != 200:
                            raise MLServiceError(f"Sentiment service error: {resp.status}")
                        data = await resp.json()
                        return data.get("predictions", [])  # type: ignore[no-any-return]
    except MLServiceError:
        raise
    except Exception as exc:
        logger.warning("sentiment.service_unavailable", error=str(exc))
        return []
    return []


def _aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    if not scores:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    totals: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for score in scores:
        label = str(score.get("label", "neutral")).lower()
        conf = float(score.get("score", 0.0))
        if label in totals:
            totals[label] += conf

    n = len(scores)
    return {k: round(v / n, 4) for k, v in totals.items()}
