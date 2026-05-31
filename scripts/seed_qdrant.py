"""Seed Qdrant with SEC filings for ~10 large-cap tickers.

Usage:
    uv run python scripts/seed_qdrant.py
    uv run python scripts/seed_qdrant.py --ticker AAPL --ticker MSFT
    uv run python scripts/seed_qdrant.py --dry-run
    uv run python scripts/seed_qdrant.py --reset   # wipes data/raw, Qdrant, Postgres chunks
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiohttp
import structlog
import typer

logger = structlog.get_logger()

SEED_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "JPM", "V", "WMT"]

# Known-good AAPL 10-K URL used to verify SEC archive access before the main loop.
_SEC_PROBE_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
)

app = typer.Typer()


@app.command()
def main(
    tickers: list[str] = typer.Option(SEED_TICKERS, "--ticker", "-t"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    reset: bool = typer.Option(
        False, "--reset", help="Clear all stored filings and re-ingest from scratch."
    ),
) -> None:
    """Download SEC filings and ingest them into Qdrant + Postgres."""
    asyncio.run(_seed(tickers, dry_run, reset))


async def _verify_sec_access(user_agent: str) -> None:
    """Hit a known-good EDGAR URL to confirm the URL pattern and User-Agent are accepted."""
    headers = {"User-Agent": user_agent}
    async with (
        aiohttp.ClientSession() as session,
        session.get(
            _SEC_PROBE_URL,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp,
    ):
        logger.info("seed.sec_probe", url=_SEC_PROBE_URL, status=resp.status)
        if resp.status != 200:
            raise SystemExit(
                f"SEC archive probe returned HTTP {resp.status} — "
                "URL pattern or User-Agent rejected. Aborting before fetching tickers."
            )


async def _reset_stores(settings: object) -> None:  # type: ignore[type-arg]
    """Delete data/raw marker files, drop+recreate Qdrant collection, truncate Postgres chunks."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    from finsight.config import Settings  # type: ignore[attr-defined]
    from finsight.rag.stores import PostgresStore, QdrantStore  # type: ignore[attr-defined]

    assert isinstance(settings, Settings)

    raw_dir = Path("data/raw")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        logger.info("seed.reset.raw_deleted", path=str(raw_dir))
    raw_dir.mkdir(parents=True, exist_ok=True)

    qdrant = QdrantStore(settings)
    existing = await qdrant._client.get_collections()
    names = [c.name for c in existing.collections]
    if settings.qdrant_collection in names:
        await qdrant._client.delete_collection(settings.qdrant_collection)
        logger.info("seed.reset.qdrant_collection_deleted", name=settings.qdrant_collection)
    await qdrant.ensure_collection()

    pg = PostgresStore(settings)
    async with AsyncSession(pg._engine) as session:
        await session.execute(text("DROP TABLE IF EXISTS document_chunks CASCADE"))
        await session.execute(
            text(
                """
                CREATE TABLE document_chunks (
                    chunk_id         TEXT PRIMARY KEY,
                    text             TEXT NOT NULL,
                    ticker           TEXT NOT NULL,
                    form_type        TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    filing_date      TEXT NOT NULL,
                    section          TEXT,
                    search_vector    tsvector
                )
                """
            )
        )
        await session.execute(
            text("CREATE INDEX idx_dc_search_vector ON document_chunks USING GIN (search_vector)")
        )
        await session.execute(text("CREATE INDEX idx_dc_ticker ON document_chunks (ticker)"))
        await session.commit()
    logger.info("seed.reset.postgres_recreated")


async def _seed(tickers: list[str], dry_run: bool, reset: bool) -> None:
    from finsight.config import get_settings
    from finsight.ingestion.sec_edgar import SecEdgarFetcher
    from finsight.rag.ingest import ingest_documents

    settings = get_settings()
    logger.info("seed.start", tickers=tickers, dry_run=dry_run, reset=reset)

    await _verify_sec_access(settings.sec_edgar_user_agent)

    if reset and not dry_run:
        logger.info("seed.reset.start")
        await _reset_stores(settings)
        logger.info("seed.reset.done")

    fetcher = SecEdgarFetcher(settings)
    for ticker in tickers:
        logger.info("seed.ticker", ticker=ticker)
        filings = await fetcher.fetch_latest_filings(ticker)
        if not dry_run:
            for filing in filings:
                try:
                    await ingest_documents([filing], settings)
                    fetcher.mark_ingested(filing)
                    logger.info(
                        "seed.filing.ingested",
                        ticker=filing.ticker,
                        accession=filing.accession_number,
                    )
                except Exception as exc:
                    logger.error(
                        "seed.filing.ingest_failed",
                        ticker=filing.ticker,
                        accession=filing.accession_number,
                        error=str(exc),
                    )
        logger.info("seed.ticker.done", ticker=ticker, count=len(filings))

    logger.info("seed.complete", total_tickers=len(tickers))


if __name__ == "__main__":
    app()
