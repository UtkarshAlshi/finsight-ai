{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\froman\fcharset0 Times-Roman;\f2\froman\fcharset0 Times-Bold;
\f3\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c0\c0\c84706;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # FinSight AI \'97 Build Bootstrap Prompt\
\
You are Claude Code. You are going to build a production-grade, agentic \
financial intelligence platform called FinSight AI. Read this entire prompt \
before writing anything. Do not ask clarifying questions about scope, stack, \
or structure \'97 every decision is already made below. Only ask if a tool/API \
key is missing at runtime.\
\
## My context (so you make the right tradeoffs)\
\
I am a backend SWE with 2 years at a financial services firm (Java/Spring \
Boot/Kafka/Airflow/GCP). I am targeting SDE1/L3 roles at Google, Meta, \
Nutanix, etc. for August 2026. This project's purpose is twofold: \
(a) demonstrate end-to-end production AI/ML engineering on my resume, and \
(b) be something I can defend in deep technical interviews. Therefore: \
production-grade code quality, real evaluation, real observability, real \
tests. No toy code. No mock data where real APIs exist.\
\
## What you are building\
\
A multi-agent financial intelligence platform. A user asks complex questions \
like "Compare META vs GOOGL margin trajectories post-2022 layoffs and \
forecast Q3 sentiment" and receives a cited, grounded answer produced by \
specialized agents over a hybrid RAG layer, classical ML services, and \
tiered LLM routing.\
\
## Hard architectural decisions (do not deviate)\
\
- **Language:** Python 3.11. Use `uv` for dependency management.\
- **API framework:** FastAPI with async everywhere. SSE for streaming.\
- **Agent framework:** LangGraph (not LangChain agents, not CrewAI).\
- **LLM clients:** Anthropic SDK primary, OpenAI SDK secondary. Abstract \
  behind a `LLMClient` protocol so swapping is one line.\
- **Vector DB:** Qdrant (run locally via Docker for dev).\
- **Structured DB:** Postgres 16 with pgvector extension (for hybrid retrieval).\
- **Cache:** Redis 7 (semantic cache + rate limiting + session state).\
- **Streaming/queues:** Kafka via `aiokafka` (Redpanda for local dev \'97 \
  lighter than full Kafka).\
- **Batch orchestration:** Airflow 2.9, DAGs in `airflow/dags/`.\
- **Embeddings:** `BAAI/bge-large-en-v1.5` via sentence-transformers; \
  cross-encoder re-ranker `BAAI/bge-reranker-large`.\
- **Observability:** Langfuse for LLM tracing, OpenTelemetry for service \
  traces, structured JSON logs via `structlog`, Prometheus metrics.\
- **Eval framework:** RAGAS + a custom LLM-as-judge harness.\
- **Containerization:** Docker + docker-compose for local; Kubernetes \
  manifests in `deploy/k8s/` for GKE later.\
- **Testing:** pytest + pytest-asyncio + httpx for API tests. Aim for \
  meaningful tests, not coverage theater.\
- **Linting/formatting:** ruff + mypy (strict mode on `src/`).\
- **Pre-commit:** ruff, mypy, pytest-fast subset.\
\
## Repository structure (create exactly this)\
\
\pard\pardeftab720\sa240\partightenfactor0

\f1 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 finsight-ai/ \uc0\u9500 \u9472 \u9472  README.md \u9500 \u9472 \u9472  ARCHITECTURE.md \u9500 \u9472 \u9472  ROADMAP.md \u9500 \u9472 \u9472  pyproject.toml \u9500 \u9472 \u9472  uv.lock \u9500 \u9472 \u9472  .env.example \u9500 \u9472 \u9472  .gitignore \u9500 \u9472 \u9472  .pre-commit-config.yaml \u9500 \u9472 \u9472  docker-compose.yml \u9500 \u9472 \u9472  Makefile \u9500 \u9472 \u9472  src/finsight/ \u9474  \u9500 \u9472 \u9472  
\f2\b init
\f1\b0 .py \uc0\u9474  \u9500 \u9472 \u9472  config.py # pydantic-settings, env-driven \u9474  \u9500 \u9472 \u9472  api/ # FastAPI app \u9474  \u9474  \u9500 \u9472 \u9472  main.py \u9474  \u9474  \u9500 \u9472 \u9472  routes/ \u9474  \u9474  \u9474  \u9500 \u9472 \u9472  query.py # POST /query, SSE streaming \u9474  \u9474  \u9474  \u9500 \u9472 \u9472  health.py \u9474  \u9474  \u9474  \u9492 \u9472 \u9472  admin.py \u9474  \u9474  \u9492 \u9472 \u9472  middleware.py # request id, logging, rate limit \u9474  \u9500 \u9472 \u9472  agents/ # LangGraph nodes + graph \u9474  \u9474  \u9500 \u9472 \u9472  graph.py # the main orchestrator graph \u9474  \u9474  \u9500 \u9472 \u9472  research.py \u9474  \u9474  \u9500 \u9472 \u9472  analyst.py \u9474  \u9474  \u9500 \u9472 \u9472  sentiment.py \u9474  \u9474  \u9500 \u9472 \u9472  forecasting.py \u9474  \u9474  \u9500 \u9472 \u9472  critic.py \u9474  \u9474  \u9492 \u9472 \u9472  state.py # typed graph state \u9474  \u9500 \u9472 \u9472  llm/ \u9474  \u9474  \u9500 \u9472 \u9472  client.py # LLMClient protocol + impls \u9474  \u9474  \u9500 \u9472 \u9472  router.py # tiered model routing \u9474  \u9474  \u9500 \u9472 \u9472  cache.py # Redis semantic cache \u9474  \u9474  \u9492 \u9472 \u9472  guardrails.py # injection detection, PII, disclaimers \u9474  \u9500 \u9472 \u9472  rag/ \u9474  \u9474  \u9500 \u9472 \u9472  ingest.py # chunking + embedding pipeline \u9474  \u9474  \u9500 \u9472 \u9472  retriever.py # hybrid: BM25 + dense + rerank \u9474  \u9474  \u9500 \u9472 \u9472  chunker.py # hierarchical/semantic chunking \u9474  \u9474  \u9492 \u9472 \u9472  stores.py # Qdrant + pg wrappers \u9474  \u9500 \u9472 \u9472  ingestion/ \u9474  \u9474  \u9500 \u9472 \u9472  sec_edgar.py # 10-K/10-Q/8-K fetcher \u9474  \u9474  \u9500 \u9472 \u9472  earnings_calls.py \u9474  \u9474  \u9500 \u9472 \u9472  news_stream.py # Kafka consumer \u9474  \u9474  \u9492 \u9472 \u9472  market_data.py \u9474  \u9500 \u9472 \u9472  ml/ \u9474  \u9474  \u9500 \u9472 \u9472  sentiment_service.py # FinBERT FastAPI sub-app \u9474  \u9474  \u9500 \u9472 \u9472  forecast_service.py # Prophet/LSTM \u9474  \u9474  \u9492 \u9472 \u9472  anomaly_service.py # isolation forest \u9474  \u9500 \u9472 \u9472  eval/ \u9474  \u9474  \u9500 \u9472 \u9472  ragas_runner.py \u9474  \u9474  \u9500 \u9472 \u9472  llm_judge.py \u9474  \u9474  \u9492 \u9472 \u9472  datasets/ # eval question bank \u9474  \u9492 \u9472 \u9472  obs/ \u9474  \u9500 \u9472 \u9472  tracing.py # OTel + Langfuse \u9474  \u9492 \u9472 \u9472  metrics.py # Prometheus \u9500 \u9472 \u9472  airflow/ \u9474  \u9492 \u9472 \u9472  dags/ \u9474  \u9500 \u9472 \u9472  sec_filings_dag.py \u9474  \u9492 \u9472 \u9472  earnings_calls_dag.py \u9500 \u9472 \u9472  tests/ \u9474  \u9500 \u9472 \u9472  unit/ \u9474  \u9500 \u9472 \u9472  integration/ \u9474  \u9492 \u9472 \u9472  eval/ \u9500 \u9472 \u9472  notebooks/ # exploration only, not prod \u9500 \u9472 \u9472  deploy/ \u9474  \u9500 \u9472 \u9472  docker/ \u9474  \u9474  \u9500 \u9472 \u9472  api.Dockerfile \u9474  \u9474  \u9492 \u9472 \u9472  ml.Dockerfile \u9474  \u9492 \u9472 \u9472  k8s/ # placeholder manifests \u9500 \u9472 \u9472  scripts/ \u9474  \u9500 \u9472 \u9472  seed_qdrant.py \u9474  \u9492 \u9472 \u9472  run_eval.py \u9492 \u9472 \u9472  docs/ \u9500 \u9472 \u9472  adr/ # Architecture Decision Records \u9492 \u9472 \u9472  runbooks/\
\pard\pardeftab720\qc\partightenfactor0

\f3\fs22 \cf3 \strokec3 \
\
## Build order (execute in this order, commit after each step)\
\
### Step 0 \'97 Repo skeleton\
- Initialize git, create the full directory tree, add `.gitignore` \
  (Python, IDE, env, data dirs).\
- Create `pyproject.toml` with `uv`, all deps pinned to current stable \
  versions. Include dev deps (ruff, mypy, pytest, pre-commit).\
- Write `README.md` with: project pitch (3 lines), architecture diagram \
  placeholder, quickstart (3 commands), tech stack table, link to \
  ARCHITECTURE.md and ROADMAP.md.\
- Write `ARCHITECTURE.md` with: system context diagram (mermaid), component \
  responsibilities, data flow for a sample query, sequence diagram for the \
  agent graph (mermaid), key tradeoffs section.\
- Write `ROADMAP.md` with the 4-month phased plan (see "Phases" below).\
- Write `.env.example` with every env var the system reads, with comments.\
- Write `Makefile` with: `make up`, `make down`, `make test`, `make lint`, \
  `make eval`, `make ingest`.\
- Write `docker-compose.yml` with services: postgres+pgvector, qdrant, \
  redis, redpanda, langfuse, the api service, the ml service. Healthchecks \
  on every service. Named volumes for data.\
- Set up pre-commit and run it once.\
- Commit: `chore: bootstrap repo skeleton and tooling`.\
\
### Step 1 \'97 Config, logging, observability foundation\
- `src/finsight/config.py`: pydantic-settings, all config via env, typed.\
- `src/finsight/obs/`: structlog setup with request-id propagation, \
  Langfuse client init, OTel tracer, Prometheus registry.\
- A minimal FastAPI app that exposes `/health`, `/metrics`, and emits a \
  test trace + log on a `/ping` endpoint.\
- Tests proving health and metrics work.\
- Commit: `feat(obs): config, structured logging, tracing, metrics`.\
\
### Step 2 \'97 LLM client + router + semantic cache + guardrails\
- `LLMClient` protocol with `generate`, `stream`, `embed`.\
- Anthropic and OpenAI implementations behind it.\
- Tiered router: cheap model for classification/lookup, premium for \
  synthesis. Routing decision is a function of an enum `TaskComplexity`.\
- Redis semantic cache: embed query, cosine-search recent queries, \
  return cached answer if similarity > threshold. Configurable TTL.\
- Guardrails module: prompt-injection heuristics, PII regex redaction, \
  required financial-advice disclaimer appended to user-facing answers.\
- Unit tests for routing logic, cache hit/miss, guardrail triggers.\
- Commit: `feat(llm): tiered routing, semantic cache, guardrails`.\
\
### Step 3 \'97 Ingestion: SEC EDGAR (start narrow)\
- `ingestion/sec_edgar.py`: async fetcher for a configurable list of \
  tickers, downloads latest 10-K and last 4 10-Qs, parses with \
  `sec-parser` or BeautifulSoup, stores raw + structured to Postgres \
  and raw files to a local `data/raw/` (gitignored).\
- One Airflow DAG that runs this daily.\
- Idempotency: don't re-download or re-process unchanged filings (use \
  filing accession number as the key).\
- Tests with VCR-recorded fixtures (do not hit SEC in CI).\
- Commit: `feat(ingest): SEC EDGAR pipeline with Airflow DAG`.\
\
### Step 4 \'97 RAG: chunking, embedding, hybrid retrieval\
- `rag/chunker.py`: hierarchical chunking that respects 10-K section \
  boundaries (Item 1, Item 1A, Item 7, etc.). Fallback to semantic \
  chunking for free-form text.\
- `rag/ingest.py`: chunk \uc0\u8594  embed \u8594  upsert into Qdrant with rich metadata \
  (ticker, filing_type, section, fiscal_period, chunk_id).\
- `rag/retriever.py`: hybrid retrieval = BM25 (Postgres `tsvector`) + \
  dense (Qdrant) \uc0\u8594  reciprocal rank fusion \u8594  cross-encoder rerank \u8594  \
  top-k. All async.\
- `scripts/seed_qdrant.py`: ingests filings for ~10 large-cap tickers \
  (AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, JPM, V, WMT) so the \
  system is demoable out of the box.\
- Tests: retrieval quality on a small held-out set, latency budget.\
- Commit: `feat(rag): hybrid retrieval with reranking`.\
\
### Step 5 \'97 Single-agent end-to-end (vertical slice)\
- A minimal LangGraph with one node (research agent) + critic node.\
- Wire it to: retriever \uc0\u8594  LLM (with cache) \u8594  critic \u8594  response.\
- POST `/query` endpoint, SSE streaming, returns answer + citations \
  (chunk metadata, not just URLs).\
- Integration test: ask "What were Apple's reported revenues in the \
  most recent 10-Q?" and verify citation correctness.\
- Commit: `feat(api): vertical slice \'97 single-agent RAG with citations`.\
\
### Step 6 \'97 Multi-agent graph (Phase 1 MVP)\
- Add analyst, sentiment, forecasting, critic agents.\
- Router node decides which agents to invoke based on parsed query intent.\
- Shared typed `GraphState` with messages, retrieved_context, \
  ml_outputs, citations, draft_answer, validation_result.\
- Each agent has tool access (retriever, SQL exec, ML service calls).\
- Critic loops up to N times if groundedness fails.\
- Mermaid diagram of the graph in ARCHITECTURE.md, kept in sync.\
- Tests for each agent in isolation + one happy-path E2E.\
- Commit: `feat(agents): multi-agent LangGraph orchestrator`.\
\
### Step 7 \'97 Classical ML services\
- Three FastAPI sub-apps under `src/finsight/ml/`, each a deployable \
  service, each callable as an agent tool.\
- Sentiment: FinBERT inference on earnings call snippets.\
- Forecasting: Prophet baseline on historical price/revenue series.\
- Anomaly: isolation forest on volume.\
- Each service: `/predict` endpoint, model versioning, latency metric.\
- Commit: `feat(ml): sentiment, forecast, anomaly services`.\
\
### Step 8 \'97 Evaluation harness\
- A curated eval dataset of 50 questions in `eval/datasets/` with \
  ground-truth answers and required citation tickers/sections.\
- `eval/ragas_runner.py`: faithfulness, answer relevancy, context \
  precision, context recall.\
- `eval/llm_judge.py`: a separate LLM scores answers on rubric \
  (correctness, completeness, citation quality).\
- `scripts/run_eval.py`: runs full eval, writes a markdown report \
  with metric trends to `docs/eval_reports/`.\
- Make this a CI job that runs nightly (GitHub Actions stub).\
- Commit: `feat(eval): RAGAS + LLM judge + nightly eval`.\
\
### Step 9 \'97 Polish\
- Add request rate limiting (Redis token bucket).\
- Add circuit breakers around external calls (`circuitbreaker` lib).\
- Cost tracking: per-request token + dollar accounting, exposed as \
  Prometheus metric.\
- A small Next.js dashboard in `web/` (separate package) that calls \
  the SSE endpoint and renders streamed answers + citations + \
  agent-trace timeline. Keep it minimal, functional, dark-themed.\
- Write 2 ADRs in `docs/adr/`: "Why LangGraph over LangChain" and \
  "Why hybrid retrieval over pure dense".\
- Write 1 runbook in `docs/runbooks/`: "Investigating a hallucination \
  regression".\
- Commit: `feat: rate limit, circuit breakers, cost metrics, web ui`.\
\
## Phases (write into ROADMAP.md, do not implement Phases 2\'964 now)\
\
- **Phase 1 (Steps 0\'969 above):** Working end-to-end system, evaluable, \
  demoable.\
- **Phase 2:** Add news Kafka stream, multi-modal table extraction from \
  PDFs (vision model), expand to 50 tickers.\
- **Phase 3:** QLoRA fine-tune Llama 3.1 8B on 50K distilled financial \
  Q&A pairs, deploy via vLLM, A/B test against base model in eval harness.\
- **Phase 4:** Deploy to GCP GKE, autoscaling, blue/green, write the \
  Medium post on semantic caching cost wins.\
\
## Conventions you must follow\
\
- **Type hints everywhere.** mypy strict on `src/`. No `Any` without \
  a comment justifying it.\
- **Async everywhere** that does I/O. No sync calls inside async paths.\
- **No secrets in code.** Everything via env, surfaced through `config.py`.\
- **Errors are typed.** Custom exception hierarchy in \
  `src/finsight/errors.py`. API maps them to HTTP codes in middleware.\
- **Logs are structured.** Never `print`. Always `logger.info("event", \
  key=value)`.\
- **Tests live next to features.** When you add a module, add its tests \
  in the matching `tests/` path in the same commit.\
- **Commits are small and conventional.** `feat:`, `fix:`, `chore:`, \
  `docs:`, `test:`, `refactor:`. One logical change per commit. Do not \
  bundle Step 4 and Step 5 into one commit.\
- **Every external API call goes through a thin client wrapper** with \
  retries (tenacity), timeouts, and a circuit breaker. No raw `httpx` \
  scattered around.\
- **Document as you build.** When you finish a step, update \
  ARCHITECTURE.md and README.md if anything user-facing changed. Do \
  not leave docs for the end.\
\
## What you should NOT do\
\
- Do not invent fictional libraries or APIs. If unsure, check pypi.\
- Do not write 500-line files. Split aggressively.\
- Do not add features not listed above. If you think one is needed, \
  note it in ROADMAP.md under "Considered" and move on.\
- Do not skip tests because "it's just a prototype." It is not.\
- Do not generate placeholder/stub code that silently returns mock \
  data. If a step requires a real API key I have not provided, stop \
  and ask me for it.\
\
## Start now\
\
1. Confirm you have read this prompt by printing a one-paragraph \
   summary of the architecture in your own words.\
2. List the env vars / API keys you will need from me (Anthropic, \
   OpenAI, SEC EDGAR user-agent string, Langfuse keys, etc.).\
3. Then begin Step 0.}