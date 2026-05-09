"""Redis semantic cache: embed query → cosine similarity → cache hit/miss."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
import structlog
from redis.asyncio import Redis
from sentence_transformers import SentenceTransformer

from finsight.obs.metrics import cache_hits_total, cache_misses_total

logger = structlog.get_logger()

_CACHE_PREFIX = "finsight:semantic_cache:"
_INDEX_KEY = "finsight:cache_index"

# Type alias for index entries
_IndexEntry = dict[str, Any]


class SemanticCache:
    """Embed user queries and cache LLM responses by semantic similarity.

    On each lookup: embed query → scan recent entries → return response if
    cosine similarity > threshold. Falls through on miss.
    """

    def __init__(
        self,
        redis: Redis[bytes],
        model_name: str,
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_index_size: int = 1000,
    ) -> None:
        self._redis = redis
        self._model = SentenceTransformer(model_name)
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_index = max_index_size

    async def get(self, query: str) -> str | None:
        """Return cached answer if a semantically similar query exists."""
        query_vec = self._embed(query)
        index = await self._load_index()

        best_sim = 0.0
        best_key: str | None = None
        for entry in index:
            stored_vec = np.array(entry["vec"], dtype=np.float32)
            sim = float(np.dot(query_vec, stored_vec))
            if sim > best_sim:
                best_sim = sim
                best_key = str(entry["key"])

        if best_sim >= self._threshold and best_key:
            cached = await self._redis.get(best_key)
            if cached:
                cache_hits_total.inc()
                logger.info("cache.hit", similarity=round(best_sim, 4))
                return cached.decode()
            # Entry expired; fall through
        cache_misses_total.inc()
        logger.debug("cache.miss", best_similarity=round(best_sim, 4))
        return None

    async def set(self, query: str, answer: str) -> None:
        """Store query embedding and answer in cache."""
        query_vec = self._embed(query)
        key = _CACHE_PREFIX + hashlib.sha256(query.encode()).hexdigest()[:16]

        await self._redis.set(key, answer, ex=self._ttl)

        index = await self._load_index()
        index.append({"key": key, "vec": query_vec.tolist(), "ts": time.time()})
        # Trim index to max size (evict oldest)
        if len(index) > self._max_index:
            index = sorted(index, key=lambda e: float(e["ts"]))[-self._max_index :]
        await self._redis.set(_INDEX_KEY, json.dumps(index), ex=self._ttl * 2)
        logger.debug("cache.stored", key=key)

    async def _load_index(self) -> list[_IndexEntry]:
        raw = await self._redis.get(_INDEX_KEY)
        if not raw:
            return []
        try:
            data: list[_IndexEntry] = json.loads(raw)
            return data
        except json.JSONDecodeError:
            return []

    def _embed(self, text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)
