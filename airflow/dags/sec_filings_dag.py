"""Airflow DAG: daily SEC EDGAR filing ingestion for configured tickers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from airflow.decorators import dag, task  # type: ignore[import-untyped]

DEFAULT_ARGS = {
    "owner": "finsight",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="sec_filings_ingest",
    description="Daily SEC EDGAR ingestion — 10-K and last 4 10-Qs per ticker",
    schedule="0 6 * * *",  # 6 AM UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ingestion", "sec", "edgar"],
)
def sec_filings_dag() -> None:
    @task()
    def fetch_filings(ticker: str) -> dict[str, int]:
        """Fetch and store filings for one ticker; returns count of new filings."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from finsight.config import get_settings
        from finsight.ingestion.sec_edgar import SecEdgarFetcher

        settings = get_settings()
        fetcher = SecEdgarFetcher(settings)
        filings = asyncio.run(fetcher.fetch_latest_filings(ticker))
        return {"ticker": ticker, "new_filings": len(filings)}

    @task()
    def get_tickers() -> list[str]:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from finsight.config import get_settings

        return get_settings().ticker_list

    tickers = get_tickers()
    fetch_filings.expand(ticker=tickers)


sec_filings_dag()
