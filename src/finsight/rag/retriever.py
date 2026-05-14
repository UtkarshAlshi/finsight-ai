"""Hybrid retrieval: BM25 (Postgres) + dense (Qdrant) → RRF → cross-encoder rerank.

Pipeline:
1. BM25 search on Postgres tsvector (exact lexical matches)
2. Dense search on Qdrant (semantic similarity)
3. Reciprocal Rank Fusion to merge ranked lists
4. Cross-encoder reranking for final top-k selection
"""

from __future__ import annotations

import time
from typing import Any

import structlog
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from finsight.config import Settings
from finsight.obs.metrics import retrieval_latency_seconds
from finsight.rag.stores import PostgresStore, QdrantStore

logger = structlog.get_logger()

_RRF_K = 60  # standard RRF constant

# Module-level singletons — loaded once at startup via warm_models(), never per-request.
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def warm_models(settings: Settings) -> None:
    """Load embedding and reranker models into module singletons; no-op if already loaded.

    Called explicitly at API startup (lifespan) so the cold-load cost is paid
    once before any request is served. HybridRetriever.__init__ also calls this
    as a safety net, making it safe to instantiate the retriever in tests without
    going through the lifespan.
    """
    global _embedder, _reranker  # noqa: PLW0603
    if _embedder is not None:
        return
    device = _select_device()
    logger.info("models.loading", device=device, embedding_model=settings.embedding_model)
    _embedder = SentenceTransformer(settings.embedding_model, device=device)
    _reranker = CrossEncoder(settings.reranker_model, device=device)
    logger.info(
        "models.loaded",
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )


class HybridRetriever:
    """Combines BM25 + dense retrieval with RRF fusion and cross-encoder reranking."""

    def __init__(self, settings: Settings) -> None:
        self._pg = PostgresStore(settings)
        self._qdrant = QdrantStore(settings)
        # warm_models is a no-op when called at startup; this fallback handles
        # direct instantiation in tests and scripts.
        warm_models(settings)
        assert _embedder is not None and _reranker is not None
        self._embedder = _embedder
        self._reranker = _reranker
        self._retrieval_k = settings.retrieval_top_k
        self._rerank_k = settings.rerank_top_k

    async def retrieve(
        self,
        query: str,
        *,
        ticker: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run full hybrid retrieval pipeline and return reranked chunks."""
        final_k = top_k or self._rerank_k

        # Step 1: BM25
        t0 = time.perf_counter()
        bm25_results = await self._pg.bm25_search(query, top_k=self._retrieval_k, ticker=ticker)
        retrieval_latency_seconds.labels(stage="bm25").observe(time.perf_counter() - t0)
        logger.debug("retrieval.bm25", count=len(bm25_results))

        # Step 2: Dense
        t1 = time.perf_counter()
        query_vec = self._embedder.encode(query, normalize_embeddings=True).tolist()
        filter_: dict[str, Any] | None = {"ticker": ticker} if ticker else None
        dense_results = await self._qdrant.search(
            query_vector=query_vec,
            top_k=self._retrieval_k,
            filter_=filter_,
        )
        retrieval_latency_seconds.labels(stage="dense").observe(time.perf_counter() - t1)
        logger.debug("retrieval.dense", count=len(dense_results))

        # Step 3: RRF fusion
        t2 = time.perf_counter()
        fused = _reciprocal_rank_fusion(bm25_results, dense_results, k=_RRF_K)
        retrieval_latency_seconds.labels(stage="rrf").observe(time.perf_counter() - t2)

        # Step 4: Cross-encoder rerank
        t3 = time.perf_counter()
        reranked = self._rerank(query, fused, top_k=final_k)
        retrieval_latency_seconds.labels(stage="rerank").observe(time.perf_counter() - t3)
        logger.info("retrieval.complete", query_len=len(query), returned=len(reranked))

        return reranked

    def _rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        texts = [c.get("text", "") for c in candidates]
        pairs = [[query, t] for t in texts]
        scores = self._reranker.predict(pairs)
        scored = sorted(
            zip(candidates, scores, strict=False), key=lambda x: float(x[1]), reverse=True
        )
        return [item for item, _ in scored[:top_k]]


def _reciprocal_rank_fusion(
    list_a: list[dict[str, Any]],
    list_b: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i) for each list containing the document.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(list_a, start=1):
        doc_id = str(doc.get("chunk_id", doc.get("_id", rank)))
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = doc

    for rank, doc in enumerate(list_b, start=1):
        doc_id = str(doc.get("chunk_id", doc.get("_id", rank)))
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [docs[doc_id] for doc_id in sorted_ids]
