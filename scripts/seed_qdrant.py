"""Seed Qdrant with SEC filings for ~10 large-cap tickers."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog
import typer

logger = structlog.get_logger()

SEED_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "JPM", "V", "WMT"]

app = typer.Typer()


@app.command()
def main(
    tickers: list[str] = typer.Option(SEED_TICKERS, "--ticker", "-t"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Download SEC filings and ingest them into Qdrant + Postgres."""
    asyncio.run(_seed(tickers, dry_run))


async def _seed(tickers: list[str], dry_run: bool) -> None:
    # Imports deferred so this script can be imported without all deps installed
    from finsight.config import get_settings
    from finsight.ingestion.sec_edgar import SecEdgarFetcher
    from finsight.rag.ingest import ingest_documents

    settings = get_settings()
    logger.info("seed.start", tickers=tickers, dry_run=dry_run)

    fetcher = SecEdgarFetcher(settings)
    for ticker in tickers:
        logger.info("seed.ticker", ticker=ticker)
        documents = await fetcher.fetch_latest_filings(ticker)
        if not dry_run:
            await ingest_documents(documents, settings)
        logger.info("seed.ticker.done", ticker=ticker, count=len(documents))

    logger.info("seed.complete", total_tickers=len(tickers))


if __name__ == "__main__":
    app()
