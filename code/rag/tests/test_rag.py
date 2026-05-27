"""
test_rag.py
-----------
Unit tests for the CherryPicker RAG package. These tests intentionally do NOT
require Chroma, an OpenAI API key, or any LLM provider — they cover the pure
logic pieces (citation parsing, parameter validation, prompt assembly) so they
can run anywhere with `pytest`.

Run from the project root:
    pip install pytest
    pytest code/rag/tests/ -q
"""

import pytest

from code.rag.answer   import _parse_cited_ranks, _split_citations
from code.rag.prompts  import build_rag_prompt, SYSTEM_INSTRUCTION
from code.rag.retrieve import RetrievedChunk, search


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------

def test_parse_citations_basic():
    assert _parse_cited_ranks("A [1]. B [2,4]. C [1].") == [1, 2, 4]


def test_parse_citations_with_spaces_and_repeats():
    txt = "Foo [3, 5]. Bar [3]. Baz [ 7 , 8 ]. [3,5,7]."
    assert _parse_cited_ranks(txt) == [3, 5, 7, 8]


def test_parse_citations_empty():
    assert _parse_cited_ranks("No citations here at all.") == []


def test_parse_citations_ignores_malformed():
    # [abc] and [1.5] should be skipped, but [2] still works
    assert _parse_cited_ranks("[abc] [1.5] [2]") == [2]


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

def test_split_citations_partitions_valid_and_invalid():
    valid, invalid = _split_citations([1, 2, 9, 4, 12], n_chunks=5)
    assert valid == [1, 2, 4]
    assert invalid == [9, 12]


def test_split_citations_all_valid():
    valid, invalid = _split_citations([3, 1, 2], n_chunks=3)
    assert valid == [3, 1, 2]
    assert invalid == []


def test_split_citations_all_invalid():
    valid, invalid = _split_citations([0, 99], n_chunks=5)
    assert valid == []
    assert invalid == [0, 99]


# ---------------------------------------------------------------------------
# Retrieval parameter validation
# These all raise BEFORE any Chroma access, so they don't need a populated DB.
# ---------------------------------------------------------------------------

def test_search_rejects_empty_query():
    with pytest.raises(ValueError, match="empty query"):
        search("")


def test_search_rejects_whitespace_query():
    with pytest.raises(ValueError, match="empty query"):
        search("   \n\t")


def test_search_rejects_bad_lambda():
    with pytest.raises(ValueError, match="lambda_mmr"):
        search("abc", lambda_mmr=1.5)
    with pytest.raises(ValueError, match="lambda_mmr"):
        search("abc", lambda_mmr=-0.1)


def test_search_rejects_fetch_n_less_than_k():
    with pytest.raises(ValueError, match="fetch_n"):
        search("abc", k=10, fetch_n=5)


def test_search_rejects_k_below_one():
    with pytest.raises(ValueError, match="k must be"):
        search("abc", k=0)


def test_search_rejects_unknown_source():
    with pytest.raises(ValueError, match="source must be"):
        search("abc", source="BAD")


def test_search_accepts_valid_sources():
    # These calls will get past validation and may then raise FileNotFoundError
    # if Chroma isn't built locally. We just want to confirm validation passes,
    # so we accept anything OTHER than ValueError("source ...").
    for src in (None, "FT", "MO"):
        try:
            search("abc", source=src)
        except ValueError as e:
            assert "source must be" not in str(e), \
                f"source={src!r} should pass validation, got: {e}"
        except Exception:
            pass  # FileNotFoundError from Chroma, etc. — fine for this test


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _chunk(rank: int, **kw) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for prompt tests."""
    defaults = dict(
        chunk_id=f"id-{rank}",
        text=f"text body {rank}",
        score=0.9 - 0.1 * rank,
        year=2024,
        source="FT",
        pmcid=f"PMC{1000 + rank}",
    )
    defaults.update(kw)
    return RetrievedChunk(**defaults)


def test_prompt_contains_question_and_sources_header():
    chunks = [_chunk(1), _chunk(2)]
    system, user = build_rag_prompt("What is the question?", chunks)
    assert system == SYSTEM_INSTRUCTION
    assert "Question:" in user
    assert "What is the question?" in user
    assert "Sources:" in user


def test_prompt_numbers_chunks_starting_at_one():
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    _, user = build_rag_prompt("Q?", chunks)
    assert "[1]" in user
    assert "[2]" in user
    assert "[3]" in user
    # We shouldn't see [0] or [4] for 3 chunks
    assert "[0]" not in user
    assert "[4]" not in user


def test_prompt_includes_provenance_in_each_chunk_header():
    chunks = [_chunk(1, year=2023, pmcid="PMC123", source="MO")]
    _, user = build_rag_prompt("Q?", chunks)
    assert "2023" in user
    assert "PMC123" in user
    assert "MO" in user


def test_prompt_handles_no_chunks():
    _, user = build_rag_prompt("Q?", [])
    assert "No sources retrieved" in user


def test_prompt_unknown_style_raises():
    with pytest.raises(NotImplementedError):
        build_rag_prompt("Q?", [_chunk(1)], style="json-structured")
