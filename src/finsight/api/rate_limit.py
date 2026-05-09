"""Redis token-bucket rate limiter for API endpoints.

Each client (identified by IP or API key) gets a bucket of `capacity` tokens
that refills at `rate` tokens/second. A request costs 1 token.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from redis.asyncio import Redis

from finsight.errors import RateLimitError

logger = structlog.get_logger()

_BUCKET_TTL = 3600  # Redis key expiry in seconds


class TokenBucketRateLimiter:
    """Redis-backed token bucket; all state lives in Redis so it's multi-process safe."""

    def __init__(self, redis: Redis[bytes], capacity: int = 60, rate: float = 1.0) -> None:
        self._redis = redis
        self._capacity = capacity
        self._rate = rate

    async def check(self, key: str) -> None:
        """Raise RateLimitError if the bucket for `key` is empty."""
        bucket_key = f"rl:{key}"
        now = time.time()

        raw: dict[bytes, bytes] = await self._redis.hgetall(bucket_key) or {}

        tokens = float(raw.get(b"tokens", self._capacity))
        last_refill = float(raw.get(b"last_refill", now))

        # Refill tokens based on elapsed time
        elapsed = now - last_refill
        tokens = min(self._capacity, tokens + elapsed * self._rate)

        if tokens < 1.0:
            logger.warning("rate_limit.exceeded", key=key)
            raise RateLimitError(f"Rate limit exceeded for {key}")

        tokens -= 1.0

        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.hset(bucket_key, mapping={"tokens": tokens, "last_refill": now})
            await pipe.expire(bucket_key, _BUCKET_TTL)
            await pipe.execute()

        logger.debug("rate_limit.ok", key=key, remaining=round(tokens, 1))


def get_client_key(request: Any) -> str:
    """Extract rate-limit key from request — IP with X-Forwarded-For support."""
    forwarded: str | None = request.headers.get("X-Forwarded-For")
    if forwarded:
        return str(forwarded.split(",")[0].strip())
    if request.client:
        return str(request.client.host)
    return "unknown"
