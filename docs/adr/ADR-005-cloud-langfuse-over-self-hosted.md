# ADR-005: Cloud Langfuse Instead of Self-Hosted

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** FinSight AI team

## Context

Langfuse is used for LLM trace observability (prompt/completion logging,
latency tracking, cost attribution). The original docker-compose included
a `langfuse-server` service (image: `langfuse/langfuse:2`) backed by the
same Postgres instance. This added a stateful web service to the local
stack, consuming memory and adding startup time.

During Phase 1 development, the self-hosted Langfuse container was
observed to be a frequent source of startup failures (slow healthcheck,
database migration conflicts) that blocked `make up` without being
related to the system under development.

Additionally, Langfuse Cloud's free tier covers the trace volume of a
portfolio project. A self-hosted instance in a local dev stack provides
no meaningful advantage over the cloud offering at this scale.

## Decision

Remove `langfuse-server` from docker-compose. Configure the API container
to point to Langfuse Cloud (`https://cloud.langfuse.com`) via
`LANGFUSE_HOST` in `.env`.

The `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`
environment variables remain in `.env.example` — only the host value
changes from `http://localhost:3000` to the cloud endpoint.

## Consequences

**Positive:**
- `make up` is faster and more reliable — 5 services instead of 6
- Langfuse UI is always available via browser without a local container
- No risk of Langfuse migration failures blocking local development

**Negative:**
- Traces are sent to a third-party service; not suitable if the data
  being traced is confidential (not a concern for SEC public filings)
- Requires a Langfuse Cloud account; the free tier is sufficient for
  this project but is an external dependency

## Alternatives Considered

| Option | Rejected Because |
|--------|-----------------|
| Keep self-hosted, fix healthcheck | Spent meaningful time debugging Langfuse startup; the fix was not worth the maintenance cost |
| Remove Langfuse entirely | LLM trace observability is a stated project requirement; it demonstrates the observability stack |
| Use OpenTelemetry only | OTel covers infrastructure traces but not LLM-specific prompt/completion logging |
