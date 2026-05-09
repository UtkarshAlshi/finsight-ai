"""Unit tests for rate limiter, circuit breaker, and cost tracker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.errors import CircuitOpenError, RateLimitError

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_starts_closed() -> None:
    from finsight.llm.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test")
    assert not cb.is_open


def test_circuit_breaker_opens_after_threshold() -> None:
    from finsight.llm.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open
    cb.record_failure()
    assert cb.is_open


def test_circuit_breaker_check_raises_when_open() -> None:
    from finsight.llm.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=1)
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.check()


def test_circuit_breaker_closes_on_success() -> None:
    from finsight.llm.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=1)
    cb.record_failure()
    assert cb.is_open
    cb.record_success()
    assert not cb.is_open


def test_circuit_breaker_half_open_after_timeout() -> None:
    import time

    from finsight.llm.circuit_breaker import CircuitBreaker, _State

    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    # Accessing is_open transitions to HALF_OPEN
    assert not cb.is_open
    assert cb._state == _State.HALF_OPEN


def test_get_breaker_returns_same_instance() -> None:
    from finsight.llm.circuit_breaker import get_breaker

    b1 = get_breaker("shared-service")
    b2 = get_breaker("shared-service")
    assert b1 is b2


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


def test_cost_tracker_known_model() -> None:
    from finsight.llm.cost_tracker import RequestCost

    cost = RequestCost(model="claude-haiku-4-5-20251001", input_tokens=1000, output_tokens=500)
    assert cost.usd > 0.0
    # 1000 * 0.80/1M + 500 * 4.00/1M = 0.0008 + 0.002 = 0.0028
    assert abs(cost.usd - 0.0028) < 1e-6


def test_cost_tracker_unknown_model_uses_sonnet_pricing() -> None:
    from finsight.llm.cost_tracker import _estimate_cost

    cost_unknown = _estimate_cost("unknown-model", 1000, 500)
    cost_sonnet = _estimate_cost("claude-sonnet-4-6", 1000, 500)
    assert cost_unknown == cost_sonnet


def test_record_usage_increments_prometheus() -> None:
    from finsight.llm.cost_tracker import record_usage

    with (
        patch("finsight.llm.cost_tracker.llm_tokens_total") as mock_tokens,
        patch("finsight.llm.cost_tracker.llm_cost_usd_total") as mock_cost,
    ):
        mock_tokens.labels.return_value = MagicMock()
        mock_cost.labels.return_value = MagicMock()

        result = record_usage(
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            input_tokens=500,
            output_tokens=200,
        )

    assert result.input_tokens == 500
    assert result.output_tokens == 200
    assert result.usd > 0.0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_allows_request() -> None:
    from finsight.api.rate_limit import TokenBucketRateLimiter

    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={})
    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=False)
    mock_pipe.execute = AsyncMock(return_value=[1, 1])
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    limiter = TokenBucketRateLimiter(mock_redis, capacity=10, rate=1.0)
    # Should not raise
    await limiter.check("test-client")


@pytest.mark.asyncio
async def test_rate_limiter_blocks_empty_bucket() -> None:
    import time

    from finsight.api.rate_limit import TokenBucketRateLimiter

    mock_redis = AsyncMock()
    # Bucket has 0 tokens, refill time is now (no elapsed time)
    mock_redis.hgetall = AsyncMock(
        return_value={b"tokens": b"0.0", b"last_refill": str(time.time()).encode()}
    )

    limiter = TokenBucketRateLimiter(mock_redis, capacity=10, rate=1.0)
    with pytest.raises(RateLimitError):
        await limiter.check("blocked-client")


def test_get_client_key_from_forwarded_header() -> None:
    from finsight.api.rate_limit import get_client_key

    mock_request = MagicMock()
    mock_request.headers = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    mock_request.client = None

    key = get_client_key(mock_request)
    assert key == "203.0.113.5"


def test_get_client_key_falls_back_to_client_host() -> None:
    from finsight.api.rate_limit import get_client_key

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "192.168.1.100"

    key = get_client_key(mock_request)
    assert key == "192.168.1.100"
