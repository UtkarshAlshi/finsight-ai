"""Unit tests for SEC EDGAR fetcher — no live network (mocked responses)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finsight.config import Settings
from finsight.ingestion.sec_edgar import Filing, SecEdgarFetcher


@pytest.fixture
def settings() -> Settings:
    return Settings(sec_edgar_user_agent="FinSightAI test@example.com")


@pytest.fixture
def fetcher(settings: Settings, tmp_path: Path) -> SecEdgarFetcher:
    f = SecEdgarFetcher(settings)
    f._raw_dir = tmp_path  # redirect writes to temp directory
    return f


def _make_submissions_payload(ticker: str = "AAPL") -> dict[str, object]:
    return {
        "cik": "0000320193",
        "tickers": [ticker],
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-Q", "10-Q", "10-Q", "8-K"],
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000100",
                    "0000320193-23-000090",
                    "0000320193-23-000080",
                    "0000320193-23-000070",
                    "0000320193-24-000050",
                ],
                "filingDate": [
                    "2024-11-01",
                    "2024-08-02",
                    "2024-05-03",
                    "2024-02-02",
                    "2023-11-03",
                    "2024-10-01",
                ],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240629.htm",
                    "aapl-20240330.htm",
                    "aapl-20231230.htm",
                    "aapl-20230930.htm",
                    "8k.htm",
                ],
            }
        },
    }


def test_extract_filing_metas_returns_correct_forms(fetcher: SecEdgarFetcher) -> None:
    submissions = _make_submissions_payload()
    metas = fetcher._extract_filing_metas(submissions, "AAPL", "0000320193")  # type: ignore[arg-type]

    forms = [m.form_type for m in metas]
    assert forms.count("10-K") == 1
    assert forms.count("10-Q") == 4
    assert "8-K" not in forms


def test_extract_filing_metas_limits_10q_to_four(fetcher: SecEdgarFetcher) -> None:
    submissions = _make_submissions_payload()
    metas = fetcher._extract_filing_metas(submissions, "AAPL", "0000320193")  # type: ignore[arg-type]
    assert sum(1 for m in metas if m.form_type == "10-Q") == 4


def test_already_stored_returns_false_for_new(fetcher: SecEdgarFetcher) -> None:
    assert not fetcher._already_stored("0000320193-24-999999")


def test_already_stored_returns_true_after_save(fetcher: SecEdgarFetcher, tmp_path: Path) -> None:
    filing = Filing(
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
        document_url="https://example.com/doc.htm",
        raw_text="Some text content here.",
        sections={"item_1_business": "We sell phones."},
    )
    fetcher._save_raw(filing)
    assert fetcher._already_stored("0000320193-24-000123")


def test_parse_html_strips_tags(fetcher: SecEdgarFetcher) -> None:
    html = "<html><body><p>Revenue grew <b>12%</b></p><script>x=1</script></body></html>"
    text = fetcher._parse_html(html)
    assert "Revenue grew" in text
    assert "<script>" not in text
    assert "<b>" not in text


def test_extract_10k_sections_finds_mda(fetcher: SecEdgarFetcher) -> None:
    text = (
        "Some preamble\n"
        "Item 7. Management's Discussion and Analysis\n"
        "Revenue was strong.\n"
        "Item 8. Financial Statements\n"
        "Balance sheet data."
    )
    sections = fetcher._extract_10k_sections(text)
    assert "item_7_mda" in sections
    assert "Revenue was strong." in sections["item_7_mda"]


def test_save_raw_creates_json_file(fetcher: SecEdgarFetcher, tmp_path: Path) -> None:
    filing = Filing(
        ticker="MSFT",
        cik="0000789019",
        accession_number="0000789019-24-000001",
        form_type="10-K",
        filing_date="2024-07-30",
        document_url="https://example.com/msft.htm",
        raw_text="Microsoft text",
        sections={},
    )
    fetcher._save_raw(filing)
    saved = tmp_path / "0000789019_24_000001.json"
    assert saved.exists()
    data = json.loads(saved.read_text())
    assert data["ticker"] == "MSFT"
    assert data["form_type"] == "10-K"
