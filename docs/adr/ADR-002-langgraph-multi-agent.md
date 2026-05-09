# ADR-002: LangGraph Multi-Agent Orchestration

**Status:** Accepted
**Date:** 2026-05-03
**Deciders:** FinSight AI team

## Context

Financial questions require multiple specialized capabilities: document
retrieval, SQL-based ratio analysis, earnings call sentiment, and revenue
forecasting. These can be computed independently, but their outputs must be
synthesised into a coherent answer that is then validated for groundedness.

We need an orchestration framework that supports:
- Parallel execution of independent agents
- State propagation between nodes
- Conditional loops (critic → re-synthesise)
- Streaming output

## Decision

Use **LangGraph** (`StateGraph`) with the following topology:

```
router → agents (parallel) → synthesise → critic → END
                                  ↑           |
                                  └───────────┘ (loop ≤3)
```

**Nodes:**
- `router`: classifies intent, sets routing flags (currently pass-through)
- `agents`: runs `asyncio.gather` over research/analyst/sentiment/forecast
- `synthesise`: premium LLM merges all outputs into a draft answer
- `critic`: cheap LLM PASS/FAIL groundedness check; auto-passes at iteration 3

**State:** `GraphState` (Pydantic BaseModel) carries question, retrieved
context, draft answer, ML outputs, citations, and iteration count.

### Why LangGraph over plain asyncio?

- Declarative graph topology makes agent relationships explicit and auditable
- Built-in state management via `StateGraph` avoids manual state-passing code
- `CompiledStateGraph.ainvoke()` handles checkpointing, interrupts, and streaming
- Conditional edges encode the critic loop without ad-hoc control flow

### Why loop back to `synthesise` not `research`?

Re-retrieval on each critic failure would be expensive (retrieval + embedding
overhead) and unnecessary when the underlying documents are already in
`retrieved_context`. The critic's FAIL verdict indicates poor synthesis quality,
not missing evidence, so re-synthesising with the existing context is the
correct response.

## Consequences

**Positive:**
- Parallel agents reduce wall-clock latency (4 agents → ~1× latency of slowest)
- Explicit graph topology is easy to test and reason about
- Critic loop improves answer groundedness without user interaction

**Negative:**
- LangGraph adds a framework dependency; its API has changed between versions
- Graph compilation must happen at startup (cold-start cost)
- Debugging multi-agent state requires understanding LangGraph internals

## Alternatives Considered

| Option | Rejected Because |
|--------|-----------------|
| Sequential LLM chain | No parallelism; 4× latency |
| Plain asyncio gather | Manual state management; no conditional loops |
| CrewAI | Less control over state schema; higher abstraction |
| Autogen | Heavier; designed for multi-turn agent conversations |
