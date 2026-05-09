"""Guardrails: prompt-injection detection, PII redaction, financial disclaimer."""

from __future__ import annotations

import re

import structlog

from finsight.errors import GuardrailError

logger = structlog.get_logger()

FINANCIAL_DISCLAIMER = (
    "\n\n---\n*This is AI-generated financial analysis for informational purposes only. "
    "It does not constitute financial advice. Always consult a qualified financial advisor "
    "before making investment decisions.*"
)

# ── Prompt injection heuristics ───────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+|your\s+)?instructions?", re.IGNORECASE
    ),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"disregard (your |all |previous )?", re.IGNORECASE),
    re.compile(
        r"(forget|bypass|override) (your |the |all )?(system|instructions?|rules?)", re.IGNORECASE
    ),
    re.compile(r"act as (an?|the) ", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN mode", re.IGNORECASE),
]

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"\b\d{16}\b"), "[CARD REDACTED]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL REDACTED]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE REDACTED]"),
]


def check_prompt_injection(text: str) -> None:
    """Raise GuardrailError if injection attempt is detected."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("guardrail.injection_detected", pattern=pattern.pattern[:50])
            raise GuardrailError("Prompt injection detected. Request rejected.")


def redact_pii(text: str) -> str:
    """Redact PII patterns from text and return the sanitised version."""
    redacted = text
    for pattern, replacement in _PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    if redacted != text:
        logger.info("guardrail.pii_redacted")
    return redacted


def append_disclaimer(answer: str) -> str:
    """Append the required financial advice disclaimer to user-facing answers."""
    return answer + FINANCIAL_DISCLAIMER


def apply_all(user_input: str) -> str:
    """Full guardrail pipeline for incoming user input.

    Returns sanitised input or raises GuardrailError.
    """
    check_prompt_injection(user_input)
    return redact_pii(user_input)
