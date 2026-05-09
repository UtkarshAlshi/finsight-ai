.PHONY: up down test lint eval ingest fmt check migrate seed

# ── Infrastructure ────────────────────────────────────────────
up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	docker compose ps

down:
	docker compose down

# ── Development ───────────────────────────────────────────────
install:
	uv sync --all-extras

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint:
	uv run ruff check src/ tests/
	uv run mypy src/finsight/

check: lint
	uv run pytest tests/unit/ -m "not slow" --tb=short

# ── Testing ───────────────────────────────────────────────────
test:
	uv run pytest tests/ -m "not eval" -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v -m integration

test-cov:
	uv run pytest tests/unit/ --cov=src/finsight --cov-report=html --cov-report=term-missing

# ── Database ──────────────────────────────────────────────────
migrate:
	uv run alembic upgrade head

# ── Data & Seeding ────────────────────────────────────────────
ingest:
	uv run python scripts/seed_qdrant.py

# ── Evaluation ────────────────────────────────────────────────
eval:
	uv run python scripts/run_eval.py

# ── Helpers ───────────────────────────────────────────────────
logs:
	docker compose logs -f api ml

shell:
	docker compose exec api bash

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
