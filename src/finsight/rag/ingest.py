"""RAG ingestion pipeline: chunk → embed → upsert into Qdrant + Postgres."""

from __future__ import annotations

import numpy as np
import structlog
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from finsight.config import Settings
from finsight.ingestion.sec_edgar import Filing
from finsight.rag.chunker import Chunk, chunk_filing
from finsight.rag.stores import PostgresStore, QdrantStore

logger = structlog.get_logger()

_BATCH_SIZE = 32


class EmbeddingService:
    """Wraps SentenceTransformer for batch embedding with normalisation."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in np.array(vecs)]

    def embed_one(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return list(np.array(vec).tolist())


async def ingest_documents(filings: list[Filing], settings: Settings) -> None:
    """Full pipeline: chunk → embed → upsert for a list of Filing objects."""
    if not filings:
        return

    embedding_svc = EmbeddingService(settings.embedding_model)
    qdrant = QdrantStore(settings)
    pg = PostgresStore(settings)

    await qdrant.ensure_collection()

    for filing in filings:
        chunks = chunk_filing(
            text=filing.raw_text,
            sections=filing.sections,
            ticker=filing.ticker,
            form_type=filing.form_type,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Process in batches
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            await _upsert_batch(batch, embedding_svc, qdrant, pg)

        logger.info(
            "ingest.filing_complete",
            ticker=filing.ticker,
            form=filing.form_type,
            accession=filing.accession_number,
            total_chunks=len(chunks),
        )


async def _upsert_batch(
    chunks: list[Chunk],
    embedding_svc: EmbeddingService,
    qdrant: QdrantStore,
    pg: PostgresStore,
) -> None:
    texts = [c.text for c in chunks]
    vectors = embedding_svc.embed_batch(texts)

    points: list[PointStruct] = []
    for chunk, vec in zip(chunks, vectors, strict=True):
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "ticker": chunk.ticker,
            "form_type": chunk.form_type,
            "accession_number": chunk.accession_number,
            "filing_date": chunk.filing_date,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
        }
        points.append(PointStruct(id=chunk.chunk_id, vector=vec, payload=payload))

        await pg.store_chunk(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "ticker": chunk.ticker,
                "form_type": chunk.form_type,
                "accession_number": chunk.accession_number,
                "filing_date": chunk.filing_date,
                "section": chunk.section,
            }
        )

    await qdrant.upsert(points)
