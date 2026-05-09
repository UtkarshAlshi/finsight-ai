"""Unit tests for hybrid retriever — RRF fusion logic (no DB calls)."""

from finsight.rag.retriever import _reciprocal_rank_fusion


def _doc(chunk_id: str, text: str = "text") -> dict[str, str]:
    return {"chunk_id": chunk_id, "text": text}


def test_rrf_merges_unique_docs() -> None:
    list_a = [_doc("a"), _doc("b"), _doc("c")]
    list_b = [_doc("d"), _doc("e"), _doc("f")]
    result = _reciprocal_rank_fusion(list_a, list_b)
    ids = {r["chunk_id"] for r in result}
    assert ids == {"a", "b", "c", "d", "e", "f"}


def test_rrf_boosts_doc_appearing_in_both_lists() -> None:
    list_a = [_doc("a"), _doc("b"), _doc("c")]
    list_b = [_doc("a"), _doc("d"), _doc("e")]
    result = _reciprocal_rank_fusion(list_a, list_b)
    # "a" appears first in both lists → should rank first after RRF
    assert result[0]["chunk_id"] == "a"


def test_rrf_empty_inputs() -> None:
    assert _reciprocal_rank_fusion([], []) == []


def test_rrf_one_empty_list() -> None:
    list_a = [_doc("x"), _doc("y")]
    result = _reciprocal_rank_fusion(list_a, [])
    assert len(result) == 2


def test_rrf_preserves_order_when_no_overlap() -> None:
    list_a = [_doc("a"), _doc("b")]
    list_b = [_doc("c"), _doc("d")]
    result = _reciprocal_rank_fusion(list_a, list_b)
    # a and c are both rank-1 in their lists; their RRF score equals 1/(60+1)
    # Only ordering guarantee: a and c before b and d
    ids = [r["chunk_id"] for r in result]
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("d")
