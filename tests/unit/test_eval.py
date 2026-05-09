"""Unit tests for evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def test_eval_dataset_loads() -> None:
    """Eval dataset JSON is parseable and has required fields."""
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "finsight"
        / "eval"
        / "datasets"
        / "eval_questions.json"
    )
    data = json.loads(dataset_path.read_text())
    assert len(data) == 50
    for item in data:
        assert "id" in item
        assert "question" in item
        assert "ground_truth" in item
        assert "tickers" in item
        assert "category" in item


def test_eval_dataset_unique_ids() -> None:
    dataset_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "finsight"
        / "eval"
        / "datasets"
        / "eval_questions.json"
    )
    data = json.loads(dataset_path.read_text())
    ids = [item["id"] for item in data]
    assert len(ids) == len(set(ids)), "Duplicate IDs in eval dataset"


def test_judge_parse_scores_valid() -> None:
    from finsight.eval.judge import _parse_scores

    raw = "ACCURACY: 4\nCOMPLETENESS: 3\nCITATIONS: 5"
    scores = _parse_scores(raw)
    assert scores["accuracy"] == 4
    assert scores["completeness"] == 3
    assert scores["citations"] == 5
    assert scores["composite"] == pytest.approx(12 / 15, abs=1e-3)


def test_judge_parse_scores_missing_fields() -> None:
    from finsight.eval.judge import _parse_scores

    raw = "ACCURACY: 3"
    scores = _parse_scores(raw)
    assert scores["accuracy"] == 3
    assert scores["completeness"] == 0
    assert scores["citations"] == 0


def test_judge_empty_answer_returns_zeros() -> None:
    from finsight.eval.judge import _parse_scores

    scores = _parse_scores("")
    assert scores["accuracy"] == 0
    assert scores["completeness"] == 0
    assert scores["citations"] == 0
    assert scores["composite"] == 0.0


@pytest.mark.asyncio
async def test_judge_answer_calls_llm() -> None:
    from finsight.config import Settings
    from finsight.eval.judge import judge_answer

    settings = Settings()
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value="ACCURACY: 4\nCOMPLETENESS: 4\nCITATIONS: 3")

    with patch("finsight.eval.judge.LLMRouter") as MockRouter:
        MockRouter.return_value.get_client.return_value = mock_client
        scores = await judge_answer(
            question="What was Apple revenue?",
            answer="Apple revenue was $391B in FY2024.",
            ground_truth="Apple reported total net sales of $391.0 billion in fiscal year 2024.",
            settings=settings,
        )

    assert scores["accuracy"] == 4
    assert scores["completeness"] == 4
    assert scores["citations"] == 3
    assert scores["composite"] == pytest.approx(11 / 15, abs=1e-3)


@pytest.mark.asyncio
async def test_ragas_runner_summary_structure() -> None:
    """run_eval should return a summary with expected keys."""
    from finsight.config import Settings
    from finsight.eval.ragas_runner import run_eval

    settings = Settings()

    mock_state = AsyncMock()
    mock_state.final_answer = "Apple revenue was $391B."
    mock_state.draft_answer = ""
    mock_state.retrieved_context = [{"text": "Revenue was $391B."}]

    with (
        patch("finsight.agents.graph.run_query", return_value=mock_state),
        patch(
            "finsight.eval.ragas_runner._ragas_score",
            return_value={"faithfulness": 0.9, "answer_relevancy": 0.85},
        ),
    ):
        summary = await run_eval(settings, max_questions=2)

    assert "total" in summary
    assert "answered" in summary
    assert "avg_faithfulness" in summary
    assert "avg_answer_relevancy" in summary
    assert summary["total"] == 2
