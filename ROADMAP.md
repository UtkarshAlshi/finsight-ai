# FinSight AI — Roadmap

## Phase 1 — Working End-to-End System (Steps 0–9)

**Target:** Fully demoable, evaluable platform for August 2026 job search.

| Step | Description | Status |
|---|---|---|
| 0 | Repo skeleton, tooling, docker-compose | ✅ Done |
| 1 | Config, logging, observability foundation | ⬜ |
| 2 | LLM client, tiered router, semantic cache, guardrails | ⬜ |
| 3 | SEC EDGAR ingestion + Airflow DAG | ⬜ |
| 4 | RAG: hybrid retrieval with reranking | ⬜ |
| 5 | Single-agent vertical slice (research + critic) | ⬜ |
| 6 | Multi-agent LangGraph orchestrator | ⬜ |
| 7 | Classical ML services (FinBERT, Prophet, anomaly) | ⬜ |
| 8 | RAGAS + LLM-as-judge evaluation harness | ⬜ |
| 9 | Rate limiting, circuit breakers, cost metrics, web UI | ⬜ |

---

## Phase 2 — News Stream + Multi-Modal + Scale

- Kafka consumer for real-time financial news (Benzinga / NewsAPI)
- Multi-modal table extraction from SEC PDFs using vision model
- Expand ticker universe to 50 large-caps
- Estimated timeline: 2026 Q3

---

## Phase 3 — Fine-Tuning

- QLoRA fine-tune Llama 3.1 8B on 50K distilled financial Q&A pairs
- Deploy via vLLM with quantization
- A/B test fine-tuned vs base model in eval harness
- Estimated timeline: 2026 Q4

---

## Phase 4 — GCP Production Deployment

- GKE with Autopilot, Horizontal Pod Autoscaler
- Blue/green deployments with Argo Rollouts
- Cloud SQL (Postgres) + Memorystore (Redis) + Pub/Sub (Kafka)
- Write-up: "How semantic caching cut LLM costs 60%" on Medium
- Estimated timeline: 2027 Q1

---

## Considered (not in scope)

- Real-time options chain analysis (would require paid data vendor)
- Multi-tenant SaaS with billing (out of scope for portfolio project)
- Compliance / SOC 2 controls (Phase 4+ consideration)
