# ADR-003: Embedding Device Selection — MPS on Host, CPU inside Docker

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** FinSight AI team

## Context

The embedding model (`BAAI/bge-large-en-v1.5`, ~1.3 GB) and reranker
(`BAAI/bge-reranker-large`) are self-hosted. Inference speed depends heavily
on the device backend (CPU vs MPS vs CUDA). Two distinct execution contexts
exist:

1. **Batch ingestion** (`scripts/seed_qdrant.py`) — runs on the developer's
   host machine directly (not inside Docker)
2. **Query-time inference** — runs inside the `api` Docker container

## Decision

- **Host (seed script):** auto-detect and use MPS when `torch.backends.mps.is_available()`
  returns true (Apple Silicon). This is the fast path for batch ingestion.
- **Docker container:** always use CPU. Docker Desktop on macOS does not expose
  Metal/MPS to containers. There is no GPU passthrough on macOS for containers.

The device selection in `retriever.py::_select_device()` handles this automatically:
MPS → CUDA → CPU in that priority order. Inside Docker on macOS the result is always
CPU; on the host it is MPS on Apple Silicon or CUDA on a Linux GPU host.

## Consequences

**Positive:**
- Batch ingestion on Apple Silicon is significantly faster than CPU (MPS acceleration)
- No configuration change needed — device selection is automatic
- The same code path works on CPU-only environments (Linux CI, Docker)

**Negative:**
- Query latency inside Docker on macOS is high (~60 s per query cold) because
  embedding and reranking run on CPU
- The 1.3 GB bge-large model is a poor fit for CPU serving; a smaller model
  (e.g. bge-small-en-v1.5) would be faster at the cost of retrieval quality

## Alternatives Considered

| Option | Rejected Because |
|--------|-----------------|
| Use OpenAI embeddings for serving | Per-query API cost at volume; external dependency adds latency and failure mode |
| Shrink to bge-small-en-v1.5 now | MTEB retrieval quality is measurably lower; deferred to Phase 2 after eval baseline is established |
| Run Docker with GPU on Linux | Not the Phase 1 development environment; Phase 4 addresses production GPU deployment |
