# FinSight AI — Incident Response Runbook

## Overview

This runbook covers the most common production incidents. For each scenario:
diagnose with the listed commands, then apply the resolution steps.

---

## 1. High API Latency (P99 > 5s)

**Symptoms:** Grafana `http_request_duration_seconds` p99 spike; user reports slow responses.

**Diagnose:**
```bash
# Check LLM latency breakdown
kubectl logs -l app=finsight-api --tail=100 | jq 'select(.event=="llm.generate")'

# Check retrieval latency
kubectl logs -l app=finsight-api --tail=100 | jq 'select(.event | startswith("retrieval"))'

# Check which agent is slow
kubectl logs -l app=finsight-api --tail=100 | jq 'select(.event | startswith("parallel_agent"))'
```

**Resolution:**
1. If LLM latency: check Anthropic status page; if provider issue, wait or switch to OpenAI fallback by setting `LLM_FALLBACK_PROVIDER=openai` in env.
2. If retrieval latency: check Qdrant pod CPU/memory; restart if OOM: `kubectl rollout restart deployment/qdrant`.
3. If Postgres slow: run `EXPLAIN ANALYZE` on `bm25_search` query; check `pg_stat_activity` for blocking locks.

---

## 2. Rate Limit Errors (429s)

**Symptoms:** `RateLimitError` in logs; users see 429 responses.

**Diagnose:**
```bash
# Check which IPs are hitting limits
kubectl logs -l app=finsight-api --tail=200 | jq 'select(.event=="rate_limit.exceeded")'

# Check Redis token bucket state
redis-cli HGETALL "rl:<client_ip>"
```

**Resolution:**
1. If legitimate burst traffic: temporarily increase bucket capacity via env var `RATE_LIMIT_CAPACITY=120` and restart API pods.
2. If abuse: add IP to blocklist in load balancer (Nginx/Cloudflare).
3. If Redis is down: rate limiter will throw errors; API will continue serving (fail-open). Fix Redis first.

---

## 3. LLM Circuit Breaker Open

**Symptoms:** `CircuitOpenError` in logs; all LLM responses fail.

**Diagnose:**
```bash
kubectl logs -l app=finsight-api --tail=50 | jq 'select(.event=="circuit_breaker.open")'
```

**Resolution:**
1. Check Anthropic/OpenAI API status.
2. Circuit auto-recovers after `CIRCUIT_RECOVERY_TIMEOUT` seconds (default 30s). Monitor `circuit_breaker.half_open` log event.
3. If provider is down long-term: restart API pods (clears in-process circuit state) and configure `LLM_FALLBACK_PROVIDER` to alternate provider.

---

## 4. ML Service Unavailable (FinBERT / Prophet)

**Symptoms:** `sentiment.service_unavailable` or `forecasting.service_unavailable` in logs; sentiment/forecast fields are `null` in responses.

**Diagnose:**
```bash
curl http://ml-service:8001/health
kubectl logs -l app=finsight-ml --tail=50
```

**Resolution:**
1. The agents gracefully degrade to `null` — answer synthesis continues without ML outputs.
2. Restart ML pod: `kubectl rollout restart deployment/finsight-ml`.
3. If OOM (model loading): increase memory limits in `k8s/ml-deployment.yaml`.

---

## 5. Qdrant Collection Missing

**Symptoms:** `RetrievalError` in logs; all queries return empty context.

**Diagnose:**
```bash
curl http://qdrant:6333/collections
```

**Resolution:**
1. Re-create collection: `uv run python -c "from finsight.rag.stores import QdrantStore; import asyncio; asyncio.run(QdrantStore(...).ensure_collection())"`.
2. Re-ingest documents: trigger the Airflow `sec_filings_ingestion` DAG manually.
3. If Qdrant data is corrupted: restore from latest snapshot in `qdrant-snapshots` PVC.

---

## 6. Postgres Connection Exhaustion

**Symptoms:** `asyncpg.TooManyConnectionsError`; `analyst_node` DB errors.

**Diagnose:**
```sql
SELECT count(*), state FROM pg_stat_activity GROUP BY state;
SELECT max_conn FROM pg_settings WHERE name = 'max_connections';
```

**Resolution:**
1. Short-term: restart API pods to release connections.
2. Long-term: add PgBouncer in front of Postgres (transaction-mode pooling).
3. Check for connection leaks: ensure all `AsyncSession` uses are inside `async with` blocks.

---

## 7. Ingestion DAG Failure

**Symptoms:** Airflow `sec_filings_ingestion` DAG showing failed tasks.

**Diagnose:**
```bash
# Check Airflow task logs in the UI, or:
airflow tasks logs sec_filings_ingestion fetch_filings.<ticker> <execution_date>
```

**Resolution:**
1. If SEC EDGAR rate-limited (429): increase `SEC_REQUEST_DELAY_SECONDS` in Airflow variable and re-run.
2. If DB write failure: check Postgres disk space (`df -h`).
3. Clear and re-run specific task: `airflow tasks clear sec_filings_ingestion -t fetch_filings.<ticker>`.

---

## Escalation

| Severity | Response Time | Who |
|----------|--------------|-----|
| P0 (full outage) | 15 min | On-call engineer |
| P1 (degraded, ML down) | 1 hour | On-call engineer |
| P2 (elevated errors) | 4 hours | Next business day |
