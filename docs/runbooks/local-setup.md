# Local Setup — Bring-Up Guide

Step-by-step guide for cloning and running FinSight AI locally.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | Set memory to ≥ 8 GB (see below) |
| Python | 3.11.x | Exactly 3.11; the project pins `<3.12` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| make | any | Pre-installed on macOS/Linux |

### Docker memory recommendation

The `api` container loads `bge-large-en-v1.5` (~1.3 GB) and
`bge-reranker-large` at startup. Combined with Postgres, Qdrant, Redis,
and Redpanda, the stack comfortably uses 6–7 GB at rest.

In Docker Desktop → Settings → Resources, set Memory to **8 GB minimum**.
Containers will OOM-kill silently at lower limits (see Troubleshooting).

---

## 1. Clone and install dependencies

```bash
git clone https://github.com/<your-org>/finsight-ai.git
cd finsight-ai
uv sync --all-extras
```

---

## 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set the following. Keys marked **required** must be set
before running; optional keys can be left at their defaults.

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | **Required** | Needed for LLM synthesis. Set to `sk-...` with active billing. |
| `SEC_EDGAR_USER_AGENT` | **Required** | Must be `"AppName contact@email.com"` per EDGAR policy. |
| `LANGFUSE_PUBLIC_KEY` | Optional | Get from Langfuse Cloud. Leave blank to disable tracing. |
| `LANGFUSE_SECRET_KEY` | Optional | Paired with the public key. |
| `LANGFUSE_HOST` | Optional | Default: `https://cloud.langfuse.com`. |
| `ANTHROPIC_API_KEY` | Optional | Only needed if you want to test the Anthropic fallback path. |
| `POSTGRES_DSN` | Pre-set | Default uses port 5433 (see note). |

> **Port 5433 note:** Postgres is exposed on host port 5433, not 5432,
> to avoid conflicts with a system Postgres installation. The `.env.example`
> already reflects this — do not change it unless you know your host port 5432 is free.

---

## 3. Start services

```bash
make up
```

Expected output: all 6 containers reach `healthy` status within ~60 s.

```
NAME                STATUS          PORTS
finsight-postgres   healthy         0.0.0.0:5433->5432/tcp
finsight-qdrant     healthy         0.0.0.0:6333-6334->6333-6334/tcp
finsight-redis      healthy         0.0.0.0:6379->6379/tcp
finsight-redpanda   healthy         0.0.0.0:19092->19092/tcp ...
finsight-api        healthy         0.0.0.0:8000->8000/tcp
finsight-ml         healthy         0.0.0.0:8001->8001/tcp
```

The `api` container will show `(health: starting)` for up to 30 s while
the embedding and reranker models load. This is normal.

Check logs:

```bash
make logs
# or
docker compose logs -f api
```

---

## 4. Run database migrations

```bash
make migrate
```

This runs Alembic against the Postgres container. Expect ~2 s.

---

## 5. Seed the vector store

```bash
make ingest
```

This runs `scripts/seed_qdrant.py` against AAPL and MSFT SEC filings.

**Expected runtime:** 10–20 minutes on Apple Silicon (MPS), longer on Intel CPU.
The seed script runs on the host (not inside Docker), so it uses MPS when available.

Progress is logged per filing. The script is idempotent — re-running skips
already-processed filings.

To seed additional tickers:

```bash
uv run python scripts/seed_qdrant.py --tickers GOOGL,META
```

---

## 6. First query

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were Apple'\''s main revenue drivers in fiscal 2023?", "stream": false}' \
  | python3 -m json.tool
```

**Expected cold latency: ~60 s.** This includes:
- BM25 search on Postgres
- Dense vector search on Qdrant (CPU embedding of the query)
- RRF fusion + cross-encoder reranking
- LLM synthesis via OpenAI

Subsequent identical queries hit the Redis semantic cache and return in < 1 s.

---

## 7. Verify health endpoints

```bash
curl http://localhost:8000/health          # api
curl http://localhost:8001/health          # ml service
curl http://localhost:6333/healthz         # qdrant (browser-friendly)
```

---

## Troubleshooting

### Port 5432 conflict

If `make up` fails with a Postgres bind error:

```
Error: Bind for 0.0.0.0:5432 failed: port is already allocated
```

The host port is already 5433 in `docker-compose.yml`, so this should not
happen. If it does, check whether an older version of this repo is running:

```bash
docker compose ps
docker compose down
```

### OOM kill on api container

Symptom: `api` container exits with code 137 immediately after models start loading.

Cause: Docker Desktop memory limit is too low. The bge-large model alone is ~1.3 GB;
the full api container needs ~3 GB.

Fix: Docker Desktop → Settings → Resources → Memory → set to ≥ 8 GB, then restart.

### OpenAI quota errors

Symptom: query returns an error like `openai.RateLimitError: You exceeded your current quota`.

Cause: The `OPENAI_API_KEY` does not have active billing credit.

Fix: Add a payment method and a small credit ($5–$10) at platform.openai.com.
The retrieval path (steps 1–4 in the query pipeline) works without the key;
only LLM synthesis (steps 5–7) requires it.

### Qdrant collection not found

Symptom: `qdrant_client.http.exceptions.UnexpectedResponse: 404 Not Found`

Cause: `make migrate` was skipped, or `make ingest` has not completed.

Fix: Run `make migrate` then `make ingest`. The collection is created during
the first ingest run.

### Langfuse traces not appearing

Symptom: No traces visible in Langfuse Cloud.

Cause: `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are not set, or point
to the wrong project.

The system works without Langfuse — tracing is best-effort and failures are
logged at WARN level only. Check `docker compose logs api | grep langfuse`.
