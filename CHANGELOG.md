# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-31 — Phase 1 Initial Release

### Added

**Infrastructure**
- 6-service docker-compose stack: Postgres 16 (pgvector), Qdrant 1.12, Redis 7,
  Redpanda (Kafka-compat), FastAPI API service, FastAPI ML service
- Postgres exposed on host port 5433 to avoid conflict with system Postgres
- Single uvicorn worker (`--workers 1`) to avoid loading the 1.3 GB embedding
  model multiple times in memory

**Ingestion**
- Async SEC EDGAR fetcher using canonical `company_tickers.json` for CIK resolution
  and the modern `/Archives/edgar/data/` URL pattern
- Idempotent upsert: accession number is written to `processed_filings` only after
  full successful chunk upsert into both Postgres and Qdrant
- Hierarchical chunker with section-boundary awareness for 10-K/10-Q documents
- AAPL and MSFT indexed (~3,400 chunks); remaining 8 tickers deferred to Phase 2

**Retrieval**
- Hybrid retrieval pipeline: BM25 (Postgres tsvector) + dense (Qdrant) → RRF (k=60)
  → cross-encoder reranking → top-k
- Embedding model: `BAAI/bge-large-en-v1.5` (1024-dim); MPS on host, CPU in Docker
- Reranker: `BAAI/bge-reranker-large`
- Models loaded once at API startup via FastAPI lifespan (not per request)

**Agents**
- LangGraph `StateGraph` with nodes: router → agents (parallel) → synthesise → critic
- Parallel agent execution via `asyncio.gather`: research, analyst, sentiment, forecasting
- Critic loop with auto-pass at iteration 3 to prevent infinite loops
- `GraphState` (Pydantic BaseModel) carries full pipeline state

**LLM**
- OpenAI as primary provider: `gpt-4o-mini` (cheap tier) + `gpt-4o` (premium)
- Anthropic (`claude-haiku`, `claude-sonnet`) as optional fallback
- Tiered router: `TaskComplexity` enum routes to cheap or premium model
- Redis semantic cache with cosine similarity threshold
- Prompt-injection detection and financial disclaimer injection

**Observability**
- Structured JSON logging via structlog
- OpenTelemetry tracing with Langfuse Cloud integration (OTLP exporter)
- Prometheus metrics: retrieval latency histograms, request counters
- Request-ID middleware for end-to-end tracing

**Evaluation**
- RAGAS evaluation harness with 50-question dataset
- LLM-as-judge groundedness metric
- Harness is built and unit-tested; live run with published scores is deferred

### Known Constraints

- CPU embedding inside Docker on macOS (~60 s cold query latency)
- Only AAPL and MSFT are indexed in Phase 1
- End-to-end synthesis requires an OpenAI key with active billing credit
- Analyst, sentiment, and forecasting agents return empty results for most Phase 1 queries
- RAGAS evaluation has not been run against the live system

### Architecture Decisions

- [ADR-001](docs/adr/ADR-001-hybrid-rag-retrieval.md) — Hybrid RAG with RRF
- [ADR-002](docs/adr/ADR-002-langgraph-multi-agent.md) — LangGraph multi-agent orchestration
- [ADR-003](docs/adr/ADR-003-embedding-device-tradeoffs.md) — MPS on host, CPU in Docker
- [ADR-004](docs/adr/ADR-004-scope-reduction-two-tickers.md) — Two-ticker Phase 1 scope
- [ADR-005](docs/adr/ADR-005-cloud-langfuse-over-self-hosted.md) — Cloud Langfuse

---

[0.1.0]: https://github.com/<your-org>/finsight-ai/releases/tag/v0.1.0
