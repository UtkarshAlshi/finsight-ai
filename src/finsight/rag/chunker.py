"""Hierarchical chunker respecting 10-K section boundaries.

Strategy:
1. If the document has known 10-K sections, chunk each section independently.
2. Within each section (or the full doc if sections are absent), use a sliding-window
   character chunker with configurable size and overlap.
3. Each chunk carries metadata (ticker, form_type, section, fiscal_period, chunk_id).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

_KNOWN_SECTIONS = [
    "item_1_business",
    "item_1a_risk_factors",
    "item_7_mda",
    "item_7a_quantitative",
    "item_8_financial",
]


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing."""

    chunk_id: str
    text: str
    ticker: str
    form_type: str
    accession_number: str
    filing_date: str
    section: str
    chunk_index: int


def chunk_filing(
    text: str,
    sections: dict[str, str],
    *,
    ticker: str,
    form_type: str,
    accession_number: str,
    filing_date: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Produce chunks from a filing document.

    If section boundaries are available, each section is chunked independently
    to keep semantically coherent context together.
    """
    all_chunks: list[Chunk] = []

    if sections:
        for section_name in _KNOWN_SECTIONS:
            section_text = sections.get(section_name, "")
            if not section_text.strip():
                continue
            section_chunks = _sliding_window_chunks(
                text=section_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                ticker=ticker,
                form_type=form_type,
                accession_number=accession_number,
                filing_date=filing_date,
                section=section_name,
                chunk_offset=len(all_chunks),
            )
            all_chunks.extend(section_chunks)
    else:
        # Fallback: chunk the full document as a single section
        all_chunks = _sliding_window_chunks(
            text=text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            ticker=ticker,
            form_type=form_type,
            accession_number=accession_number,
            filing_date=filing_date,
            section="full_document",
            chunk_offset=0,
        )

    logger.info(
        "chunker.done",
        ticker=ticker,
        form=form_type,
        accession=accession_number,
        sections_used=len(sections),
        total_chunks=len(all_chunks),
    )
    return all_chunks


def _sliding_window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    *,
    ticker: str,
    form_type: str,
    accession_number: str,
    filing_date: str,
    section: str,
    chunk_offset: int,
) -> list[Chunk]:
    """Split text into overlapping character-level windows."""
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    start = 0
    idx = chunk_offset
    while start < len(text):
        end = min(start + chunk_size, len(text))
        snippet = text[start:end].strip()
        if snippet:
            chunk_id = _make_chunk_id(accession_number, section, idx)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=snippet,
                    ticker=ticker,
                    form_type=form_type,
                    accession_number=accession_number,
                    filing_date=filing_date,
                    section=section,
                    chunk_index=idx,
                )
            )
            idx += 1
        start += step

    return chunks


def _make_chunk_id(accession_number: str, section: str, index: int) -> str:
    raw = f"{accession_number}:{section}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
