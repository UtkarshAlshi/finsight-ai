"""LLM-as-judge evaluator for financial answer quality.

Uses a cheap LLM to score answers on factual accuracy, completeness,
and citation quality on a 1-5 scale. Used alongside RAGAS metrics.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from finsight.config import Settings
from finsight.llm.router import LLMRouter, TaskComplexity

logger = structlog.get_logger()

_JUDGE_SYSTEM = """You are an expert financial analyst evaluating AI-generated answers.
Score the answer on three dimensions, each from 1 to 5:
1. Factual accuracy: Are numbers and facts correct?
2. Completeness: Does it fully address the question?
3. Citation quality: Are specific figures and sources cited?

Respond ONLY in this exact format (no other text):
ACCURACY: <1-5>
COMPLETENESS: <1-5>
CITATIONS: <1-5>"""

_JUDGE_PROMPT = """Question: {question}

Ground Truth: {ground_truth}

AI Answer: {answer}

Score the AI answer:"""


async def judge_answer(
    question: str,
    answer: str,
    ground_truth: str,
    settings: Settings,
) -> dict[str, Any]:
    """Return scores dict with accuracy, completeness, citations (1-5 each)."""
    if not answer.strip():
        return {"accuracy": 0, "completeness": 0, "citations": 0, "composite": 0.0}

    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.LOW)

    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )

    try:
        raw = await client.generate(prompt, system=_JUDGE_SYSTEM, max_tokens=64)
        return _parse_scores(raw)
    except Exception as exc:
        logger.warning("judge.failed", error=str(exc))
        return {"accuracy": 0, "completeness": 0, "citations": 0, "composite": 0.0}


def _parse_scores(raw: str) -> dict[str, Any]:
    def _extract(label: str) -> int:
        m = re.search(rf"{label}:\s*([1-5])", raw, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    accuracy = _extract("ACCURACY")
    completeness = _extract("COMPLETENESS")
    citations = _extract("CITATIONS")
    composite = round((accuracy + completeness + citations) / 15.0, 4)

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "citations": citations,
        "composite": composite,
    }
