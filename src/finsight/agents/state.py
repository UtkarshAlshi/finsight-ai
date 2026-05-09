"""Shared typed state for the LangGraph agent graph."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel


class GraphState(BaseModel):
    """Immutable-by-convention typed state threaded through all graph nodes."""

    model_config = {"arbitrary_types_allowed": True}

    # Core
    question: str = ""
    session_id: str | None = None

    # Retrieval
    retrieved_context: list[dict[str, Any]] = []

    # Agent outputs
    draft_answer: str = ""
    ml_outputs: dict[str, Any] = {}
    citations: list[dict[str, str]] = []

    # Critic
    validation_result: str = ""  # "pass" | "fail" | "pending"
    critic_iterations: int = 0

    # Final
    final_answer: str = ""

    # LangGraph message accumulator (for future tool-calling nodes)
    messages: Annotated[list[Any], add_messages] = []
