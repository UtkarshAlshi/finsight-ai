"""Qdrant vector store wrapper + Postgres async wrapper for BM25 retrieval."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from finsight.config import Settings
from finsight.errors import RetrievalError

logger = structlog.get_logger()

_VECTOR_SIZE = 1024  # bge-large-en-v1.5 output dimension


class QdrantStore:
    """Thin async wrapper around Qdrant for document vector storage and search."""

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("qdrant.collection_created", name=self._collection)

    async def upsert(self, points: list[PointStruct]) -> None:
        """Upsert a batch of points into the collection."""
        if not points:
            return
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        logger.debug("qdrant.upserted", count=len(points))

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filter_: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search. Returns list of payload dicts with score."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            qdrant_filter = None
            if filter_:
                conditions: list[FieldCondition] = [
                    FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_.items()
                ]
                qdrant_filter = Filter(must=conditions)  # type: ignore[arg-type]

            results = await self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                {**r.payload, "_score": r.score, "_id": str(r.id)} for r in results if r.payload
            ]
        except Exception as exc:
            raise RetrievalError(f"Qdrant search failed: {exc}") from exc


class PostgresStore:
    """Async Postgres client for BM25 full-text search via tsvector."""

    def __init__(self, settings: Settings) -> None:
        self._engine = create_async_engine(settings.postgres_dsn, echo=False)

    async def bm25_search(
        self,
        query: str,
        top_k: int = 20,
        ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25-style search using Postgres tsvector + ts_rank."""
        try:
            async with AsyncSession(self._engine) as session:
                sql = """
                    SELECT
                        chunk_id,
                        text,
                        ticker,
                        form_type,
                        accession_number,
                        filing_date,
                        section,
                        ts_rank(search_vector, plainto_tsquery('english', :query)) AS rank
                    FROM document_chunks
                    WHERE search_vector @@ plainto_tsquery('english', :query)
                    {ticker_filter}
                    ORDER BY rank DESC
                    LIMIT :top_k
                """.format(ticker_filter="AND ticker = :ticker" if ticker else "")
                params: dict[str, Any] = {"query": query, "top_k": top_k}
                if ticker:
                    params["ticker"] = ticker
                result = await session.execute(text(sql), params)
                rows = result.mappings().all()
                return [dict(r) for r in rows]
        except Exception as exc:
            raise RetrievalError(f"BM25 search failed: {exc}") from exc

    async def store_chunk(self, chunk_data: dict[str, Any]) -> None:
        """Insert or update a single chunk row."""
        sql = """
            INSERT INTO document_chunks
                (chunk_id, text, ticker, form_type,
                 accession_number, filing_date, section, search_vector)
            VALUES
                (:chunk_id, :text, :ticker, :form_type,
                 :accession_number, :filing_date, :section,
                 to_tsvector('english', :text))
            ON CONFLICT (chunk_id) DO UPDATE
                SET text = EXCLUDED.text,
                    search_vector = EXCLUDED.search_vector
        """
        async with AsyncSession(self._engine) as session:
            await session.execute(text(sql), chunk_data)
            await session.commit()
