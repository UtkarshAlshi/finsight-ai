"""Unit tests for agent graph components (no live LLM or DB calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.agents.critic import _MAX_CRITIC_ITERATIONS
from finsight.agents.state import GraphState
from finsight.config import Settings


def test_graph_state_defaults() -> None:
    state = GraphState(question="What is AAPL's revenue?")
    assert state.question == "What is AAPL's revenue?"
    assert state.draft_answer == ""
    assert state.citations == []
    assert state.validation_result == ""
    assert state.critic_iterations == 0


def test_graph_state_max_critic_constant() -> None:
    assert _MAX_CRITIC_ITERATIONS == 3


@pytest.mark.asyncio
async def test_critic_passes_at_max_iterations() -> None:
    """When critic_iterations >= max, critic should auto-pass."""
    from finsight.agents.critic import critic_node
    from finsight.config import Settings

    settings = Settings()
    state = GraphState(
        question="q",
        draft_answer="Some answer.",
        critic_iterations=3,  # at max
        retrieved_context=[{"text": "Some context"}],
    )

    result = await critic_node(state, settings)
    assert result["validation_result"] == "pass"


@pytest.mark.asyncio
async def test_research_node_builds_citations() -> None:
    """research_node should produce citations from retrieved chunks."""
    from finsight.agents.research import research_node
    from finsight.config import Settings

    settings = Settings()
    mock_chunks = [
        {
            "chunk_id": "abc",
            "ticker": "AAPL",
            "section": "item_7_mda",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000001",
            "text": "Revenue was $391B in fiscal year 2024.",
        }
    ]

    with (
        patch("finsight.agents.research.HybridRetriever") as MockRetriever,
        patch("finsight.agents.research.LLMRouter") as MockRouter,
    ):
        mock_instance = MockRetriever.return_value
        mock_instance.retrieve = AsyncMock(return_value=mock_chunks)

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value="Apple revenue was $391B in FY2024.")
        MockRouter.return_value.get_client.return_value = mock_client

        state = GraphState(question="What was Apple revenue in 2024?")
        result = await research_node(state, settings)

    assert "draft_answer" in result
    assert "citations" in result
    citations = result["citations"]
    assert isinstance(citations, list)
    assert len(citations) >= 1
    assert citations[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Router node — every node must return a non-empty dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_node_returns_nonempty_dict() -> None:
    from finsight.agents.graph import _router_node

    settings = Settings()
    state = GraphState(question="What were Apple total revenues in the most recent 10-Q?")
    result = await _router_node(state, settings)
    assert result, "router node must return a non-empty dict"
    assert "route" in result


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("What is Apple revenue in the 10-Q?", "factual"),
        ("Predict AAPL earnings next quarter", "forecast"),
        ("What is market sentiment around MSFT?", "sentiment"),
        ("Compare operating margins of AAPL and GOOG", "analysis"),
    ],
)
def test_classify_intent(question: str, expected_intent: str) -> None:
    from finsight.agents.graph import _classify_intent

    assert _classify_intent(question) == expected_intent


# ---------------------------------------------------------------------------
# Every node must return a non-empty dict for valid input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_node_returns_nonempty_dict() -> None:
    from finsight.agents.critic import critic_node

    settings = Settings()
    state = GraphState(
        question="q",
        draft_answer="Some answer.",
        critic_iterations=3,
        retrieved_context=[{"text": "context"}],
    )
    result = await critic_node(state, settings)
    assert result, "critic node must return a non-empty dict"


@pytest.mark.asyncio
async def test_research_node_returns_nonempty_dict() -> None:
    from finsight.agents.research import research_node

    settings = Settings()
    mock_chunks = [
        {
            "chunk_id": "x1",
            "ticker": "AAPL",
            "section": "item_7_mda",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000001",
            "text": "Revenue was $391B in fiscal year 2024.",
        }
    ]
    with (
        patch("finsight.agents.research.HybridRetriever") as MockRetriever,
        patch("finsight.agents.research.LLMRouter") as MockRouter,
    ):
        MockRetriever.return_value.retrieve = AsyncMock(return_value=mock_chunks)
        MockRouter.return_value.get_client.return_value.generate = AsyncMock(return_value="Answer.")

        state = GraphState(question="What was Apple revenue?")
        result = await research_node(state, settings)

    assert result, "research node must return a non-empty dict"
    assert "draft_answer" in result
    assert "citations" in result


def test_should_loop_returns_end_on_pass() -> None:
    from finsight.agents.graph import _should_loop

    state = GraphState(question="q", validation_result="pass")
    assert _should_loop(state) == "end"


def test_should_loop_returns_synthesise_on_fail() -> None:
    from finsight.agents.graph import _should_loop

    state = GraphState(question="q", validation_result="fail", critic_iterations=1)
    assert _should_loop(state) == "synthesise"


def test_should_loop_returns_end_at_max_iterations() -> None:
    from finsight.agents.graph import _should_loop

    state = GraphState(question="q", validation_result="fail", critic_iterations=3)
    assert _should_loop(state) == "end"
