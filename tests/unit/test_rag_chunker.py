"""Unit tests for the RAG chunker."""

from finsight.rag.chunker import Chunk, _sliding_window_chunks, chunk_filing


def _base_kwargs() -> dict[str, str]:
    return {
        "ticker": "AAPL",
        "form_type": "10-K",
        "accession_number": "0000320193-24-000001",
        "filing_date": "2024-11-01",
    }


def test_chunk_filing_with_sections_produces_chunks() -> None:
    sections = {"item_7_mda": "Revenue grew 12% YoY. " * 50}
    chunks = chunk_filing(
        text="",
        sections=sections,
        chunk_size=100,
        chunk_overlap=10,
        **_base_kwargs(),
    )
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.section == "item_7_mda" for c in chunks)


def test_chunk_filing_skips_empty_sections() -> None:
    sections = {"item_1_business": "Some business text.", "item_7_mda": ""}
    chunks = chunk_filing(
        text="",
        sections=sections,
        chunk_size=100,
        chunk_overlap=10,
        **_base_kwargs(),
    )
    assert all(c.section != "item_7_mda" for c in chunks)


def test_chunk_filing_fallback_for_no_sections() -> None:
    chunks = chunk_filing(
        text="A" * 500,
        sections={},
        chunk_size=100,
        chunk_overlap=10,
        **_base_kwargs(),
    )
    assert len(chunks) > 0
    assert all(c.section == "full_document" for c in chunks)


def test_sliding_window_chunk_count() -> None:
    text = "word " * 200  # 1000 chars
    chunks = _sliding_window_chunks(
        text=text,
        chunk_size=100,
        overlap=10,
        ticker="AAPL",
        form_type="10-K",
        accession_number="0000320193-24-000001",
        filing_date="2024-11-01",
        section="item_7_mda",
        chunk_offset=0,
    )
    assert len(chunks) > 5
    # First chunk should not exceed chunk_size
    assert len(chunks[0].text) <= 100


def test_chunk_ids_are_unique() -> None:
    sections = {"item_1_business": "Text " * 200}
    chunks = chunk_filing(
        text="",
        sections=sections,
        chunk_size=80,
        chunk_overlap=10,
        **_base_kwargs(),
    )
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_overlap_creates_shared_content() -> None:
    text = "ABCDE" * 40  # 200 chars
    chunks = _sliding_window_chunks(
        text=text,
        chunk_size=20,
        overlap=5,
        ticker="MSFT",
        form_type="10-Q",
        accession_number="0000789019-24-000001",
        filing_date="2024-07-30",
        section="item_1_business",
        chunk_offset=0,
    )
    # Overlap: last 5 chars of chunk n should appear at start of chunk n+1
    if len(chunks) >= 2:
        end_of_first = chunks[0].text[-5:]
        start_of_second = chunks[1].text[:5]
        assert end_of_first == start_of_second
