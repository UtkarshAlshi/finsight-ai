"""Critic agent: checks draft answer groundedness against retrieved context."""

from __future__ import annotations

import structlog

from finsight.agents.state import GraphState
from finsight.config import Settings
from finsight.llm.router import LLMRouter, TaskComplexity

logger = structlog.get_logger()

_MAX_CRITIC_ITERATIONS = 3

_CRITIC_SYSTEM = """You are a strict factual critic for financial analysis.
Your job: decide if the DRAFT ANSWER is grounded in the provided CONTEXT excerpts.
Rules:
- Reply with exactly one word: "PASS" or "FAIL"
- FAIL if the draft makes any claim not supported by the context
- FAIL if the draft contains hedging phrases like "I believe" or "likely"
- PASS only if every factual claim can be traced to a context excerpt
"""


async def critic_node(state: GraphState, settings: Settings) -> dict[str, object]:
    """LangGraph node: validate groundedness; return pass/fail verdict."""
    if state.critic_iterations >= _MAX_CRITIC_ITERATIONS:
        logger.warning("critic.max_iterations_reached")
        return {"validation_result": "pass", "final_answer": state.draft_answer}

    context_parts = [c.get("text", "") for c in state.retrieved_context[:10]]
    context_str = "\n---\n".join(context_parts)

    prompt = (
        f"CONTEXT:\n{context_str}\n\n"
        f"DRAFT ANSWER:\n{state.draft_answer}\n\n"
        "Is the draft grounded in the context? Reply PASS or FAIL only."
    )

    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.LOW)
    verdict_raw = await client.generate(prompt, system=_CRITIC_SYSTEM, max_tokens=10)
    verdict = "pass" if "PASS" in verdict_raw.upper() else "fail"

    logger.info(
        "critic.verdict",
        verdict=verdict,
        iteration=state.critic_iterations + 1,
    )

    updates: dict[str, object] = {
        "validation_result": verdict,
        "critic_iterations": state.critic_iterations + 1,
    }
    if verdict == "pass":
        updates["final_answer"] = state.draft_answer

    return updates
