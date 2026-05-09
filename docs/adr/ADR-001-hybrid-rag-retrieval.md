# ADR-001: Hybrid RAG Retrieval with Reciprocal Rank Fusion

**Status:** Accepted
**Date:** 2026-05-01
**Deciders:** FinSight AI team

## Context

Financial Q&A requires retrieving relevant SEC filing sections that may be
matched by exact keyword (e.g., "EPS", "goodwill impairment") or by semantic
meaning ("profitability trend"). Pure dense retrieval misses rare financial
terms; pure BM25 misses paraphrased concepts.

## Decision

Use a **hybrid retrieval pipeline**:

1. **BM25** via PostgreSQL `tsvector` — exact keyword match with `ts_rank`
2. **Dense** via Qdrant with `BAAI/bge-large-en-v1.5` embeddings — semantic match
3. **Reciprocal Rank Fusion** (k=60) — merges both ranked lists without requiring
   score calibration between the two systems
4. **Cross-encoder reranking** via `BAAI/bge-reranker-large` — re-scores top-20
   candidates to produce final top-k

### Why RRF over score fusion?

Score fusion requires calibrating BM25 and cosine similarity scores, which
have incompatible distributions. RRF operates on rank positions only, making
it distribution-agnostic and more robust to score scale differences.

### Why bge-large over OpenAI embeddings?

- Self-hosted: no per-embedding API cost at query time
- `bge-large-en-v1.5` achieves competitive MTEB scores vs `text-embedding-3-large`
  at a fraction of the inference cost at volume
- Keeps retrieval latency predictable (no external network dependency)

## Consequences

**Positive:**
- Higher recall for rare financial terms (BM25) and paraphrased queries (dense)
- No score calibration needed (RRF)
- Cross-encoder improves precision on the final ranked set

**Negative:**
- Two retrieval systems to operate (Postgres + Qdrant)
- Higher latency than single-system retrieval (~150ms vs ~50ms p50)
- Self-hosted model requires GPU/CPU capacity for inference

## Alternatives Considered

| Option | Rejected Because |
|--------|-----------------|
| Dense-only (Qdrant) | Misses exact financial term matches |
| BM25-only (Postgres) | Poor semantic understanding |
| OpenAI embeddings | Per-query cost at scale; external dependency |
| Linear score fusion | Requires score calibration; fragile |
