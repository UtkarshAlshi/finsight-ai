"""Async SEC EDGAR fetcher for 10-K and 10-Q filings.

Uses the EDGAR full-text search API and data.sec.gov submission endpoints.
Idempotent: filing accession number is the primary key; skips already-stored filings.
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
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/{path}"
_FORM_TYPES = {"10-K", "10-Q"}
_MAX_10Q = 4


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
                    self._save_raw(filing)
                    filings.append(filing)
                except Exception as exc:
                    logger.warning(
                        "edgar.filing_failed",
                        ticker=ticker,
                        accession=meta.accession_number,
                        error=str(exc),
                    )
            return filings

    async def _resolve_cik(self, session: aiohttp.ClientSession, ticker: str) -> str:
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "company": "",
            "CIK": ticker,
            "type": "",
            "action": "getcompany",
            "output": "atom",
        }
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(3),
                wait=wait_exponential(min=1, max=5),
                reraise=True,
            ):
                with attempt:
                    async with session.get(
                        url, params=params, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status != 200:
                            raise IngestionError(f"EDGAR CIK lookup failed: {resp.status}")
                        text = await resp.text()
                        match = re.search(r"CIK=(\d+)", text)
                        if not match:
                            raise IngestionError(f"CIK not found for ticker {ticker}")
                        cik = match.group(1).zfill(10)
                        logger.debug("edgar.cik_resolved", ticker=ticker, cik=cik)
                        return cik
        raise IngestionError(f"Failed to resolve CIK for {ticker}")

    async def _get_submissions(self, session: aiohttp.ClientSession, cik: str) -> dict[str, Any]:
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        headers = {**self._session_headers(), "Host": "data.sec.gov"}
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(3),
                wait=wait_exponential(min=1, max=5),
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

            # Stop once we have collected everything we need
            if ten_k_found and ten_q_count >= _MAX_10Q:
                accession_path = accession.replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/full-index/"
                    f"data/{cik.lstrip('0')}/{accession_path}/{doc}"
                )
                metas.append(
                    Filing(
                        ticker=ticker,
                        cik=cik,
                        accession_number=accession,
                        form_type=form,
                        filing_date=date,
                        document_url=doc_url,
                    )
                )
                break

            accession_path = accession.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/full-index/"
                f"data/{cik.lstrip('0')}/{accession_path}/{doc}"
            )
            metas.append(
                Filing(
                    ticker=ticker,
                    cik=cik,
                    accession_number=accession,
                    form_type=form,
                    filing_date=date,
                    document_url=doc_url,
                )
            )

        return metas

    async def _fetch_and_parse(self, session: aiohttp.ClientSession, filing: Filing) -> Filing:
        headers = {
            "User-Agent": self._user_agent,
            "Host": "www.sec.gov",
        }
        async with (
            self._semaphore,
            session.get(
                filing.document_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp,
        ):
            if resp.status != 200:
                raise IngestionError(f"Filing fetch failed: {resp.status} {filing.document_url}")
            content_type = resp.headers.get("Content-Type", "")
            raw = await resp.text(errors="replace")

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

    def _save_raw(self, filing: Filing) -> None:
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
        logger.debug("edgar.raw_saved", path=str(path))
