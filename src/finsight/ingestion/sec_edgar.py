"""Async SEC EDGAR fetcher for 10-K and 10-Q filings.

Uses the official data.sec.gov submission endpoints.
Idempotent: filing accession number is the primary key; skips already-stored filings.

CIK resolution uses https://www.sec.gov/files/company_tickers.json — the
official, stable bulk ticker→CIK mapping published by SEC. The file is cached
on disk (data/cache/company_tickers.json) and in process memory so subsequent
calls are free.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import structlog
from bs4 import BeautifulSoup
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from finsight.config import Settings
from finsight.errors import IngestionError

logger = structlog.get_logger()

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
# Correct archive pattern: no /full-index/ segment
EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CACHE_PATH = Path("data/cache/company_tickers.json")
_FORM_TYPES = {"10-K", "10-Q"}
_MAX_10Q = 4

# Process-level cache: {TICKER_UPPER: "0000320193"} (10-digit zero-padded CIK)
_TICKER_MAP: dict[str, str] = {}


def _build_ticker_map(raw: dict[str, Any]) -> dict[str, str]:
    """Parse company_tickers.json into {TICKER: zero-padded-CIK}."""
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw.values()
        if "ticker" in entry and "cik_str" in entry
    }


def _filing_url(cik_padded: str, accession: str, primary_doc: str) -> str:
    """Build the correct SEC Archives URL for a filing document.

    Pattern: /Archives/edgar/data/{cik_int}/{accession_no_dashes}/{doc}
    No /full-index/ segment — that path is for index files, not documents.
    """
    cik_int = cik_padded.lstrip("0") or "0"
    accession_no_dashes = accession.replace("-", "")
    return EDGAR_ARCHIVE_URL.format(
        cik_int=cik_int,
        accession=accession_no_dashes,
        doc=primary_doc,
    )


@dataclass
class Filing:
    """Represents a single SEC filing document."""

    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    document_url: str
    raw_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)


@dataclass
class TickerCikMap:
    ticker: str
    cik: str


class SecEdgarFetcher:
    """Async fetcher that downloads, parses, and stores SEC filings.

    Rate-limits to respect EDGAR fair-use policy (10 req/s max).
    """

    def __init__(self, settings: Settings) -> None:
        self._user_agent = settings.sec_edgar_user_agent
        self._raw_dir = Path("data/raw")
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(5)  # max concurrent EDGAR requests

    def _session_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    async def fetch_latest_filings(self, ticker: str) -> list[Filing]:
        """Fetch latest 10-K and last 4 10-Qs for a ticker. Returns parsed Filings."""
        async with aiohttp.ClientSession(headers=self._session_headers()) as session:
            cik = await self._resolve_cik(session, ticker)
            submissions = await self._get_submissions(session, cik)
            filing_metas = self._extract_filing_metas(submissions, ticker, cik)
            logger.info("edgar.filings_found", ticker=ticker, count=len(filing_metas))

            filings: list[Filing] = []
            for meta in filing_metas:
                if self._already_stored(meta.accession_number):
                    logger.debug("edgar.skip_existing", accession=meta.accession_number)
                    continue
                try:
                    filing = await self._fetch_and_parse(session, meta)
                    filings.append(filing)
                except Exception as exc:
                    logger.warning(
                        "edgar.filing_failed",
                        ticker=ticker,
                        accession=meta.accession_number,
                        error=str(exc),
                    )
            return filings

    async def _load_ticker_map(self, session: aiohttp.ClientSession) -> dict[str, str]:
        """Return the process-level ticker→CIK map, fetching it if needed.

        Checks the on-disk cache first; falls back to a live SEC fetch.
        Result is stored in the module-level _TICKER_MAP for the process lifetime.
        """
        global _TICKER_MAP  # noqa: PLW0603
        if _TICKER_MAP:
            return _TICKER_MAP

        # Try disk cache first
        if _CACHE_PATH.exists():
            try:
                raw = json.loads(_CACHE_PATH.read_text())
                _TICKER_MAP = _build_ticker_map(raw)
                logger.info(
                    "edgar.ticker_map_loaded_from_cache",
                    path=str(_CACHE_PATH),
                    count=len(_TICKER_MAP),
                )
                return _TICKER_MAP
            except Exception as exc:
                logger.warning("edgar.ticker_cache_corrupt", error=str(exc))

        # Fetch from SEC
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(2),
                wait=wait_exponential(min=2, max=8),
                reraise=True,
            ):
                with attempt:
                    async with session.get(
                        _COMPANY_TICKERS_URL,
                        headers={"User-Agent": self._user_agent},
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status != 200:
                            raise IngestionError(
                                f"company_tickers.json fetch failed: {resp.status}"
                            )
                        raw = await resp.json(content_type=None)

        _TICKER_MAP = _build_ticker_map(raw)
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(raw))
        logger.info(
            "edgar.ticker_map_loaded",
            source=_COMPANY_TICKERS_URL,
            count=len(_TICKER_MAP),
        )
        return _TICKER_MAP

    async def _resolve_cik(self, session: aiohttp.ClientSession, ticker: str) -> str:
        ticker_map = await self._load_ticker_map(session)
        cik = ticker_map.get(ticker.upper())
        if not cik:
            raise IngestionError(f"Ticker '{ticker}' not found in SEC company_tickers.json")
        logger.debug("edgar.cik_resolved", ticker=ticker, cik=cik)
        return cik

    async def _get_submissions(self, session: aiohttp.ClientSession, cik: str) -> dict[str, Any]:
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        headers = {**self._session_headers(), "Host": "data.sec.gov"}
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(2),
                wait=wait_exponential(min=2, max=8),
                reraise=True,
            ):
                with attempt:
                    async with session.get(
                        url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status != 200:
                            raise IngestionError(f"Submissions fetch failed: {resp.status}")
                        return await resp.json(content_type=None)  # type: ignore[no-any-return]
        raise IngestionError(f"Failed to get submissions for CIK {cik}")

    def _extract_filing_metas(
        self,
        submissions: dict[str, Any],
        ticker: str,
        cik: str,
    ) -> list[Filing]:
        recent = submissions.get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        dates: list[str] = recent.get("filingDate", [])
        primary_docs: list[str] = recent.get("primaryDocument", [])

        metas: list[Filing] = []
        ten_k_found = False
        ten_q_count = 0

        for form, accession, date, doc in zip(forms, accessions, dates, primary_docs, strict=False):
            if form not in _FORM_TYPES:
                continue
            if form == "10-K":
                if ten_k_found:
                    continue  # only latest 10-K
                ten_k_found = True
            if form == "10-Q":
                if ten_q_count >= _MAX_10Q:
                    continue
                ten_q_count += 1

            metas.append(
                Filing(
                    ticker=ticker,
                    cik=cik,
                    accession_number=accession,
                    form_type=form,
                    filing_date=date,
                    document_url=_filing_url(cik, accession, doc),
                )
            )

            if ten_k_found and ten_q_count >= _MAX_10Q:
                break

        return metas

    async def _fetch_and_parse(self, session: aiohttp.ClientSession, filing: Filing) -> Filing:
        headers = {
            "User-Agent": self._user_agent,
            "Host": "www.sec.gov",
        }
        # Hold the semaphore slot for ≥0.5 s after the response completes.
        # With semaphore=5 this caps throughput at 5/0.5 = 10 req/s globally.
        async with self._semaphore:
            async with session.get(
                filing.document_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    raise IngestionError(
                        f"Filing fetch failed: {resp.status} {filing.document_url}"
                    )
                content_type = resp.headers.get("Content-Type", "")
                raw = await resp.text(errors="replace")
            await asyncio.sleep(0.5)  # rate-limit: ≤10 req/s across 5 slots

        if "html" in content_type.lower() or raw.strip().startswith("<"):
            filing.raw_text = self._parse_html(raw)
        else:
            filing.raw_text = raw

        filing.sections = self._extract_10k_sections(filing.raw_text)
        logger.info(
            "edgar.filing_parsed",
            ticker=filing.ticker,
            form=filing.form_type,
            accession=filing.accession_number,
            text_len=len(filing.raw_text),
        )
        return filing

    def _parse_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _extract_10k_sections(self, text: str) -> dict[str, str]:
        """Best-effort extraction of standard 10-K section headings."""
        section_patterns = {
            "item_1_business": re.compile(r"item\s+1\.?\s+business", re.IGNORECASE),
            "item_1a_risk_factors": re.compile(r"item\s+1a\.?\s+risk factors", re.IGNORECASE),
            "item_7_mda": re.compile(r"item\s+7\.?\s+management.?s discussion", re.IGNORECASE),
            "item_7a_quantitative": re.compile(r"item\s+7a\.?\s+quantitative", re.IGNORECASE),
            "item_8_financial": re.compile(r"item\s+8\.?\s+financial statements", re.IGNORECASE),
        }
        sections: dict[str, str] = {}
        lines = text.split("\n")
        current_section: str | None = None
        current_lines: list[str] = []

        for line in lines:
            matched = False
            for section_name, pattern in section_patterns.items():
                if pattern.search(line):
                    if current_section and current_lines:
                        sections[current_section] = "\n".join(current_lines).strip()
                    current_section = section_name
                    current_lines = []
                    matched = True
                    break
            if not matched and current_section:
                current_lines.append(line)

        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        return sections

    def _already_stored(self, accession_number: str) -> bool:
        path = self._raw_dir / f"{accession_number.replace('-', '_')}.json"
        return path.exists()

    def mark_ingested(self, filing: Filing) -> None:
        """Write the marker file ONLY after full ingest (chunk→embed→upsert) succeeds."""
        filename = f"{filing.accession_number.replace('-', '_')}.json"
        path = self._raw_dir / filename
        data = {
            "ticker": filing.ticker,
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date,
            "document_url": filing.document_url,
            "raw_text": filing.raw_text,
            "sections": filing.sections,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug("edgar.marked_ingested", path=str(path))
