"""RAGAS evaluation runner.

Loads eval questions, runs them through the graph (or a mock), scores with
RAGAS faithfulness + answer_relevancy, and writes results to JSON.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_DATASET_PATH = Path(__file__).parent / "datasets" / "eval_questions.json"
_RESULTS_DIR = Path(__file__).parent / "results"


def _load_dataset() -> list[dict[str, Any]]:
    with _DATASET_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _ragas_score(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """Compute RAGAS metrics. Falls back to 0.0 if RAGAS not installed."""
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore[import-untyped]

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        df = result.to_pandas()
        return {
            "faithfulness": round(float(df["faithfulness"].iloc[0]), 4),
            "answer_relevancy": round(float(df["answer_relevancy"].iloc[0]), 4),
        }
    except Exception as exc:
        logger.warning("ragas.unavailable", error=str(exc))
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}


async def _run_single(
    item: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    from finsight.agents.graph import run_query

    question = item["question"]
    ground_truth = item["ground_truth"]

    start = time.perf_counter()
    try:
        state = await run_query(question, settings)
        answer = state.final_answer or state.draft_answer
        contexts = [c.get("text", "") for c in state.retrieved_context[:5]]
    except Exception as exc:
        logger.warning("eval.query_failed", id=item["id"], error=str(exc))
        answer = ""
        contexts = []
    elapsed = time.perf_counter() - start

    scores = _ragas_score(question, answer, contexts, ground_truth)

    return {
        "id": item["id"],
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts_count": len(contexts),
        "latency_seconds": round(elapsed, 3),
        **scores,
    }


async def run_eval(
    settings: Any,
    max_questions: int | None = None,
    concurrency: int = 5,
) -> dict[str, Any]:
    """Run evaluation suite; return summary dict."""
    dataset = _load_dataset()
    if max_questions is not None:
        dataset = dataset[:max_questions]

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _run_single(item, settings)

    logger.info("eval.start", questions=len(dataset))
    results = await asyncio.gather(*[_bounded(item) for item in dataset])

    answered = [r for r in results if r["answer"]]
    avg_faith = sum(r["faithfulness"] for r in answered) / len(answered) if answered else 0.0
    avg_rel = sum(r["answer_relevancy"] for r in answered) / len(answered) if answered else 0.0

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total": len(dataset),
        "answered": len(answered),
        "avg_faithfulness": round(avg_faith, 4),
        "avg_answer_relevancy": round(avg_rel, 4),
        "results": list(results),
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = _RESULTS_DIR / f"eval_{ts}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "eval.done",
        answered=len(answered),
        faithfulness=round(avg_faith, 4),
        relevancy=round(avg_rel, 4),
        output=str(out_path),
    )
    return summary


if __name__ == "__main__":
    from finsight.config import get_settings

    asyncio.run(run_eval(get_settings()))
