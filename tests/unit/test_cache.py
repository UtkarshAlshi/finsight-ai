"""Unit tests for SemanticCache — uses fakeredis to avoid a live Redis."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from finsight.llm.cache import SemanticCache


def _make_cache(threshold: float = 0.92) -> SemanticCache:
    """Build a SemanticCache with mocked Redis and embedding model."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock()

    with patch("finsight.llm.cache.SentenceTransformer") as MockModel:
        instance = MockModel.return_value
        # Returns a normalised unit vector (similarity = 1.0 with itself)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        instance.encode.return_value = vec

        cache = SemanticCache(
            redis=redis_mock,
            model_name="BAAI/bge-large-en-v1.5",
            threshold=threshold,
            ttl_seconds=60,
        )
        # Attach mocks for test inspection
        cache._redis = redis_mock  # type: ignore[attr-defined]
        cache._model = instance  # type: ignore[attr-defined]
        return cache


@pytest.mark.asyncio
async def test_cache_miss_when_empty() -> None:
    cache = _make_cache()
    cache._redis.get = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    result = await cache.get("What is Apple's revenue?")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_calls_redis_set() -> None:
    cache = _make_cache()
    await cache.set("What is Apple's revenue?", "Apple revenue is $394B.")
    assert cache._redis.set.called  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cache_hit_above_threshold() -> None:
    """When cosine similarity >= threshold, cache hit should be returned."""
    cache = _make_cache(threshold=0.5)

    stored_answer = "Apple revenue is $394B."
    vec = [1.0, 0.0, 0.0]

    # Simulate an index entry with same vector (similarity = 1.0)
    import json

    index_entry = [{"key": "finsight:semantic_cache:abc123", "vec": vec, "ts": 1.0}]
    cache._redis.get = AsyncMock(
        side_effect=[  # type: ignore[attr-defined]
            json.dumps(index_entry).encode(),  # index lookup
            stored_answer.encode(),  # value lookup
        ]
    )

    result = await cache.get("What is Apple's revenue?")
    assert result == stored_answer


@pytest.mark.asyncio
async def test_cache_miss_below_threshold() -> None:
    """When similarity < threshold, should return None."""
    cache = _make_cache(threshold=0.99)

    # Orthogonal vector → similarity = 0.0
    cache._model.encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # type: ignore[attr-defined]

    import json

    index_entry = [{"key": "finsight:semantic_cache:abc123", "vec": [0.0, 1.0, 0.0], "ts": 1.0}]
    cache._redis.get = AsyncMock(return_value=json.dumps(index_entry).encode())  # type: ignore[attr-defined]

    result = await cache.get("Completely unrelated question")
    assert result is None
