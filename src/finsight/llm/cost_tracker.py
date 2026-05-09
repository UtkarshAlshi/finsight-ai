"""Per-request LLM cost tracking.

Tracks token usage and estimated USD cost per request using known pricing.
Writes totals to Prometheus counters (already defined in obs/metrics.py) and
logs per-request cost via structlog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from finsight.obs.metrics import llm_cost_usd_total, llm_tokens_total

logger = structlog.get_logger()

# Pricing per million tokens (input / output) in USD, as of 2025-Q2
_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
}

_PER_MILLION = 1_000_000.0


@dataclass
class RequestCost:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = field(init=False)

    def __post_init__(self) -> None:
        self.usd = _estimate_cost(self.model, self.input_tokens, self.output_tokens)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _PRICING.get(model)
    if pricing is None:
        # Unknown model — use Sonnet pricing as conservative estimate
        pricing = _PRICING["claude-sonnet-4-6"]
    input_price, output_price = pricing
    cost = (input_tokens / _PER_MILLION * input_price) + (
        output_tokens / _PER_MILLION * output_price
    )
    return round(cost, 8)


def record_usage(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    session_id: str | None = None,
) -> RequestCost:
    """Record token usage to Prometheus and structlog; return cost summary."""
    cost = RequestCost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)

    llm_tokens_total.labels(model=model, provider=provider, token_type="input").inc(input_tokens)
    llm_tokens_total.labels(model=model, provider=provider, token_type="output").inc(output_tokens)
    llm_cost_usd_total.labels(model=model, provider=provider).inc(cost.usd)

    logger.info(
        "llm.cost",
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=cost.usd,
        session_id=session_id,
    )
    return cost
