# FinSight AI

A production-grade, multi-agent financial intelligence platform. Ask complex questions like *"Compare META vs GOOGL margin trajectories post-2022 and forecast Q3 sentiment"* — get cited, grounded answers produced by specialized LangGraph agents over hybrid RAG, classical ML services, and tiered LLM routing.

> **Status:** Phase 1 in progress — see [ROADMAP.md](ROADMAP.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI (SSE)                      │
│  POST /query  →  LangGraph Orchestrator              │
│                  ├── Research Agent                  │
│                  ├── Analyst Agent                   │
│                  ├── Sentiment Agent (FinBERT)        │
│                  ├── Forecasting Agent (Prophet)     │
│                  └── Critic Agent                    │
├─────────────────────────────────────────────────────┤
│  Hybrid RAG: BM25 (Postgres) + Dense (Qdrant)        │
│              → RRF fusion → Cross-encoder rerank     │
├─────────────────────────────────────────────────────┤
│  Ingestion: SEC EDGAR (Airflow DAGs)                 │
├─────────────────────────────────────────────────────┤
│  Observability: Langfuse · OTel · structlog · Prom   │
└─────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full Mermaid diagrams.

---

## Quickstart

```bash
cp .env.example .env          # fill in API keys
make up                       # start Postgres, Qdrant, Redis, Redpanda, Langfuse
make install                  # uv sync
make migrate                  # run Alembic migrations
make ingest                   # seed 10 large-cap tickers into Qdrant
uv run uvicorn src.finsight.api.main:app --reload
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115 · uvicorn · SSE |
| Agents | LangGraph 0.2 |
| LLMs | Anthropic Claude (primary) · OpenAI (secondary) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Reranker | BAAI/bge-reranker-large |
| Vector DB | Qdrant 1.12 |
| Structured DB | Postgres 16 + pgvector |
| Cache | Redis 7 (semantic cache + rate limiting) |
| Streaming | Redpanda (Kafka-compatible) via aiokafka |
| Orchestration | Airflow 2.9 |
| Observability | Langfuse · OpenTelemetry · structlog · Prometheus |
| Eval | RAGAS + LLM-as-judge |
| ML | FinBERT · Prophet · scikit-learn |
| Packaging | Python 3.11 · uv |

---

## Links

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design, component breakdown, sequence diagrams
- [ROADMAP.md](ROADMAP.md) — 4-phase plan
- [docs/adr/](docs/adr/) — Architecture Decision Records
- [docs/runbooks/](docs/runbooks/) — operational guides
