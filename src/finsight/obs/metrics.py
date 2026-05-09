"""Prometheus metrics registry and instrument definitions."""

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
)

# Use the default global registry so /metrics picks everything up automatically
registry: CollectorRegistry = REGISTRY

# ── Request metrics ───────────────────────────────────────────────────────────

http_requests_total = Counter(
    "finsight_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "finsight_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── LLM metrics ───────────────────────────────────────────────────────────────

llm_requests_total = Counter(
    "finsight_llm_requests_total",
    "Total LLM API calls",
    ["provider", "model", "task_complexity"],
)

llm_tokens_total = Counter(
    "finsight_llm_tokens_total",
    "Total tokens consumed",
    ["provider", "model", "token_type"],
)

llm_cost_usd_total = Counter(
    "finsight_llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["provider", "model"],
)

llm_latency_seconds = Histogram(
    "finsight_llm_latency_seconds",
    "LLM call latency",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# ── Cache metrics ─────────────────────────────────────────────────────────────

cache_hits_total = Counter(
    "finsight_cache_hits_total",
    "Semantic cache hits",
)

cache_misses_total = Counter(
    "finsight_cache_misses_total",
    "Semantic cache misses",
)

# ── Retrieval metrics ─────────────────────────────────────────────────────────

retrieval_latency_seconds = Histogram(
    "finsight_retrieval_latency_seconds",
    "RAG retrieval latency",
    ["stage"],
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

# ── Build info ────────────────────────────────────────────────────────────────

build_info = Info("finsight_build", "Build metadata")
build_info.info({"version": "0.1.0", "python": "3.11"})
