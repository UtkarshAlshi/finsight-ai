"""LangGraph orchestrator: research → critic loop → final answer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import partial

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finsight.agents.critic import critic_node
from finsight.agents.research import research_node
from finsight.agents.state import GraphState
from finsight.config import Settings

logger = structlog.get_logger()


def _should_loop(state: GraphState) -> str:
    """Edge function: loop back to research or exit."""
    if state.validation_result == "pass":
        return "end"
    if state.critic_iterations >= 3:
        return "end"
    return "research"


def build_graph(settings: Settings) -> CompiledStateGraph:
    """Construct and compile the research-critic LangGraph."""
    graph = StateGraph(GraphState)

    # Bind settings into nodes via partial
    graph.add_node("research", partial(research_node, settings=settings))
    graph.add_node("critic", partial(critic_node, settings=settings))

    graph.set_entry_point("research")
    graph.add_edge("research", "critic")
    graph.add_conditional_edges(
        "critic",
        _should_loop,
        {
            "end": END,
            "research": "research",
        },
    )

    return graph.compile()


async def run_query(question: str, settings: Settings, session_id: str | None = None) -> GraphState:
    """Run the full agent graph for a question and return the final state."""
    compiled = build_graph(settings)
    initial_state = GraphState(question=question, session_id=session_id)
    result = await compiled.ainvoke(initial_state.model_dump())
    return GraphState(**result)


async def stream_query(
    question: str, settings: Settings, session_id: str | None = None
) -> AsyncIterator[str]:
    """Stream the final answer token-by-token once the graph completes.

    In the vertical slice (Step 5), we run the full graph then stream the
    final answer through the LLM for the SSE response.
    """
    final_state = await run_query(question, settings, session_id)
    answer = final_state.final_answer or final_state.draft_answer

    # Stream the answer through the premium model for token-level SSE
    from finsight.llm.router import LLMRouter, TaskComplexity

    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.HIGH)

    # Re-stream the already-generated answer (avoids double LLM call in Step 5;
    # in Step 6 the streaming will come directly from agent synthesis)
    disclaimer_added = False
    async for token in client.stream(
        f"Restate this answer verbatim:\n\n{answer}",
        system="Restate the provided text exactly as given, word for word.",
    ):
        yield token

    if not disclaimer_added:
        from finsight.llm.guardrails import FINANCIAL_DISCLAIMER

        yield FINANCIAL_DISCLAIMER
