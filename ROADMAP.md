# FinSight AI — Roadmap

## Phase 1 — Working End-to-End System ✅ Complete (2026-05-31)

**Goal:** Demoable, evaluable platform with hybrid RAG, multi-agent orchestration,
and a full observability stack.

| Step | Description | Status |
|---|---|---|
| 0 | Repo skeleton, tooling, docker-compose | ✅ Done |
| 1 | Config, logging, observability foundation | ✅ Done |
| 2 | LLM client, tiered router, semantic cache, guardrails | ✅ Done |
| 3 | SEC EDGAR ingestion + idempotent upsert pipeline | ✅ Done |
| 4 | RAG: hybrid retrieval with RRF + cross-encoder reranking | ✅ Done |
| 5 | Single-agent vertical slice (research + critic) | ✅ Done |
| 6 | Multi-agent LangGraph orchestrator | ✅ Done |
| 7 | Classical ML services (FinBERT, Prophet, anomaly) | ✅ Done |
| 8 | RAGAS + LLM-as-judge evaluation harness | ✅ Done |
| 9 | Rate limiting, circuit breakers, cost metrics | ✅ Done |

### Deferred from Phase 1

These items were planned for Phase 1 but moved out deliberately to ship a
working system rather than a partially-ingested one.

| Item | Reason deferred |
|---|---|
| Full 10-ticker ingestion | CPU embedding throughput inside Docker made batch ingestion of all 10 tickers impractical. AAPL + MSFT are seeded (~3,400 chunks). See [ADR-004](docs/adr/ADR-004-scope-reduction-two-tickers.md). |
| End-to-end synthesis demo | Final LLM synthesis requires an OpenAI key with active billing credit. The retrieval path is verified; the synthesis step is blocked on billing setup. |
| Live evaluation run with published RAGAS scores | The harness is built and unit-tested. Running it against the live system requires the synthesis path to work. Deferred with the synthesis demo. |
| Web UI | Deprioritised in favour of validating the backend pipeline first. |

---

## Phase 2 — News Stream + Scale (estimated 2026 Q3)

**Goal:** Expand coverage and demonstrate real-time capabilities.

- Switch embedding model to `bge-small-en-v1.5` before ticker expansion — the
  1.3 GB bge-large model makes CPU ingestion too slow; the smaller model is
  ~4× faster with modest retrieval quality reduction
- Expand ticker universe to 10+ large-caps (unblock the Phase 1 deferrals)
- Kafka consumer for real-time financial news (Benzinga / NewsAPI)
- Multi-modal table extraction from SEC PDFs using a vision model
- Run and publish the RAGAS evaluation results
- Estimated timeline: 2026 Q3

---

## Phase 3 — Fine-Tuning (estimated 2026 Q4)

- QLoRA fine-tune Llama 3.1 8B on distilled financial Q&A pairs
- Deploy via vLLM with quantization
- A/B test fine-tuned vs base model in the eval harness
- Estimated timeline: 2026 Q4

---

## Phase 4 — GCP Production Deployment (estimated 2027 Q1)

- GKE with Autopilot, Horizontal Pod Autoscaler
- Blue/green deployments with Argo Rollouts
- Cloud SQL (Postgres) + Memorystore (Redis) + Pub/Sub (Kafka)
- Estimated timeline: 2027 Q1

---

## Considered and out of scope

- Real-time options chain analysis (requires paid data vendor)
- Multi-tenant SaaS with billing (out of scope for portfolio project)
- Compliance / SOC 2 controls (Phase 4+ consideration)
