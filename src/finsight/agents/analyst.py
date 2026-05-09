"""Analyst agent: SQL-based financial ratio analysis."""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from finsight.agents.state import GraphState
from finsight.config import Settings
from finsight.llm.router import LLMRouter, TaskComplexity

logger = structlog.get_logger()

_ANALYST_SYSTEM = """You are a financial analyst. Given the structured data below,
produce a concise analysis of financial ratios and trends. Be precise and cite figures."""

_FINANCIAL_QUERY = """
    SELECT
        ticker,
        filing_date,
        form_type,
        section,
        text
    FROM document_chunks
    WHERE ticker = ANY(:tickers)
      AND section = 'item_7_mda'
    ORDER BY filing_date DESC
    LIMIT 20
"""


async def analyst_node(state: GraphState, settings: Settings) -> dict[str, Any]:
    """LangGraph node: query structured financials and produce ratio analysis."""
    tickers = _extract_tickers(state.question)
    if not tickers:
        logger.debug("analyst.no_tickers_found")
        return {"ml_outputs": {**state.ml_outputs, "analyst": None}}

    rows = await _query_financials(settings, tickers)
    if not rows:
        return {"ml_outputs": {**state.ml_outputs, "analyst": None}}

    context = "\n\n".join(
        f"[{r['ticker']} {r['form_type']} {r['filing_date']}]\n{r['text'][:500]}" for r in rows
    )
    prompt = (
        f"Financial data for {', '.join(tickers)}:\n\n{context}\n\n"
        f"Question: {state.question}\n\n"
        "Provide ratio analysis with specific numbers:"
    )

    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.HIGH)
    analysis = await client.generate(prompt, system=_ANALYST_SYSTEM)

    logger.info("analyst.done", tickers=tickers, rows=len(rows))
    return {"ml_outputs": {**state.ml_outputs, "analyst": analysis}}


def _extract_tickers(question: str) -> list[str]:
    """Naive ticker extraction — looks for uppercase 1-5 letter words."""
    import re

    known = {"META", "GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "JPM", "V", "WMT", "GOOG"}
    found = re.findall(r"\b([A-Z]{1,5})\b", question)
    return [t for t in found if t in known]


async def _query_financials(settings: Settings, tickers: list[str]) -> list[dict[str, Any]]:
    engine = create_async_engine(settings.postgres_dsn, echo=False)
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(text(_FINANCIAL_QUERY), {"tickers": tickers})
            rows = result.mappings().all()
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("analyst.db_error", error=str(exc))
        return []
