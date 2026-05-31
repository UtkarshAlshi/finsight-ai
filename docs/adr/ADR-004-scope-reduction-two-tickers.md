# ADR-004: Phase 1 Scope Reduced to Two Tickers (AAPL + MSFT)

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** FinSight AI team

## Context

The original Phase 1 plan called for indexing 10 large-cap tickers:
AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, JPM, V, WMT. Full ingestion
requires embedding every text chunk via `bge-large-en-v1.5` (~1.3 GB model).
On CPU inside Docker, this throughput is too slow to complete in a reasonable
development session. A 10-ticker ingest ran for several hours without completing.

The alternative to waiting was to ship with partial data in an unknown state —
some tickers half-indexed, others not started.

## Decision

Ship Phase 1 with AAPL and MSFT only (~3,400 chunks). These two tickers are
fully indexed, consistent, and sufficient to demonstrate and test the full
retrieval pipeline. Queries for any indexed ticker work end-to-end.

Expanding to the full 10 tickers is straightforward:

```bash
uv run python scripts/seed_qdrant.py --tickers GOOGL,META,AMZN,NVDA,TSLA,JPM,V,WMT
```

Phase 2 will switch to `bge-small-en-v1.5` before running the expansion
(see ROADMAP.md), making CPU batch ingestion practical.

## Consequences

**Positive:**
- The shipped system is in a known-good state — fully indexed, not partially indexed
- Retrieval quality and pipeline correctness can be evaluated against real data
- Avoids shipping a "works for some tickers" demo

**Negative:**
- Cross-ticker comparison queries (e.g., AAPL vs GOOGL) will not return useful results
  until Phase 2 ingestion runs
- The 10-ticker scope stated in the original README was inaccurate until this ADR

## Alternatives Considered

| Option | Rejected Because |
|--------|-----------------|
| Ship with all 10 tickers partially ingested | Unknown data state; retrieval results unpredictable |
| Use OpenAI embeddings for batch ingest | Adds per-chunk API cost; contradicts the self-hosted embedding rationale |
| Reduce model to bge-small now | Would change the embedding dimension (384 vs 1024) mid-stream; requires re-indexing; better done cleanly in Phase 2 |
