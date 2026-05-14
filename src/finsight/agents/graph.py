"""Multi-agent LangGraph orchestrator.

Graph topology:
  router → [research, analyst, sentiment, forecasting] (parallel) → synthesise → critic → END
  critic can loop back to synthesise up to 3 times if groundedness fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finsight.agents.analyst import analyst_node
from finsight.agents.critic import critic_node
from finsight.agents.forecasting import forecasting_node
from finsight.agents.research import research_node
from finsight.agents.sentiment import sentiment_node
from finsight.agents.state import GraphState
from finsight.config import Settings

logger = structlog.get_logger()

_SYNTHESISE_SYSTEM = """You are a senior financial analyst. Synthesise the following
inputs into a coherent, well-cited answer for the analyst's question.
Use [ticker:section] inline citations. Append sentiment and forecast data if relevant."""


def _should_loop(state: GraphState) -> str:
    if state.validation_result == "pass":
        return "end"
    if state.critic_iterations >= 3:
        return "end"
    return "synthesise"


_FORECAST_KEYWORDS = {"forecast", "predict", "projection", "outlook", "guidance", "estimate"}
_SENTIMENT_KEYWORDS = {"sentiment", "news", "opinion", "analyst rating", "mood", "market reaction"}
_FACTUAL_KEYWORDS = {
    "revenue",
    "earnings",
    "eps",
    "income",
    "profit",
    "loss",
    "cash flow",
    "balance sheet",
    "10-q",
    "10-k",
    "filing",
}


def _classify_intent(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in _FORECAST_KEYWORDS):
        return "forecast"
    if any(kw in q for kw in _SENTIMENT_KEYWORDS):
        return "sentiment"
    if any(kw in q for kw in _FACTUAL_KEYWORDS):
        return "factual"
    return "analysis"


async def _router_node(state: GraphState, settings: Settings) -> dict[str, Any]:
    """Classify query intent and store routing decision in state."""
    intent = _classify_intent(state.question)
    logger.info("router.classified", intent=intent, question_len=len(state.question))
    return {"route": intent}


async def _parallel_agents_node(state: GraphState, settings: Settings) -> dict[str, Any]:
    """Run research, analyst, sentiment, forecasting concurrently."""
    results = await asyncio.gather(
        research_node(state, settings),
        analyst_node(state, settings),
        sentiment_node(state, settings),
        forecasting_node(state, settings),
        return_exceptions=True,
    )

    merged: dict[str, Any] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.warning("parallel_agent.failed", error=str(result))
        elif isinstance(result, dict):
            merged.update(result)

    return merged


async def _synthesise_node(state: GraphState, settings: Settings) -> dict[str, Any]:
    """Synthesise all agent outputs into a final draft answer."""
    from finsight.llm.router import LLMRouter, TaskComplexity

    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.HIGH)

    parts = [f"Question: {state.question}", f"\nResearch:\n{state.draft_answer}"]

    ml = state.ml_outputs
    if ml.get("analyst"):
        parts.append(f"\nFinancial Analysis:\n{ml['analyst']}")
    if ml.get("sentiment"):
        agg = ml["sentiment"].get("aggregate", {})
        parts.append(f"\nSentiment: {agg}")
    if ml.get("forecast"):
        parts.append(f"\nForecast: {ml['forecast']}")

    prompt = "\n".join(parts) + "\n\nSynthesised answer:"
    draft = await client.generate(prompt, system=_SYNTHESISE_SYSTEM)
    return {"draft_answer": draft}


def build_graph(settings: Settings) -> CompiledStateGraph:
    """Build and compile the multi-agent LangGraph."""
    graph = StateGraph(GraphState)

    graph.add_node("router", partial(_router_node, settings=settings))
    graph.add_node("agents", partial(_parallel_agents_node, settings=settings))
    graph.add_node("synthesise", partial(_synthesise_node, settings=settings))
    graph.add_node("critic", partial(critic_node, settings=settings))

    graph.set_entry_point("router")
    graph.add_edge("router", "agents")
    graph.add_edge("agents", "synthesise")
    graph.add_edge("synthesise", "critic")
    graph.add_conditional_edges(
        "critic",
        _should_loop,
        {"end": END, "synthesise": "synthesise"},
    )

    return graph.compile()


async def run_query(question: str, settings: Settings, session_id: str | None = None) -> GraphState:
    """Run the full multi-agent graph for a question; return final state."""
    compiled = build_graph(settings)
    initial = GraphState(question=question, session_id=session_id)
    result = await compiled.ainvoke(initial.model_dump())
    return GraphState(**result)


async def stream_query(
    question: str, settings: Settings, session_id: str | None = None
) -> AsyncIterator[str]:
    """Stream the final answer character-by-character once graph completes."""
    from finsight.llm.guardrails import FINANCIAL_DISCLAIMER

    final_state = await run_query(question, settings, session_id)
    answer = final_state.final_answer or final_state.draft_answer

    # Emit in 20-char chunks for SSE
    chunk_size = 20
    for i in range(0, len(answer), chunk_size):
        yield answer[i : i + chunk_size]

    yield FINANCIAL_DISCLAIMER
