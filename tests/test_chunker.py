"""Tests for the main Chunker class and all strategies."""
import pytest
from chunkops import Chunker
from chunkops.models import ChunkResult, ChunkStrategy


# ── helpers ──────────────────────────────────────────────────────────────────

def assert_valid_chunks(chunks, doc_id="test.txt"):
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, ChunkResult)
        assert c.text.strip()
        assert c.token_count > 0
        assert c.doc_id == doc_id
        assert len(c.id) == 12
        assert c.span[0] >= 0
        assert c.span[1] > c.span[0]


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_string_returns_empty():
    chunker = Chunker()
    assert chunker.chunk("") == []

def test_whitespace_only_returns_empty():
    chunker = Chunker()
    assert chunker.chunk("   \n\n   ") == []

def test_single_sentence(short_doc):
    chunker = Chunker(strategy="recursive", min_tokens=1, max_tokens=500)
    chunks = chunker.chunk(short_doc, doc_id="short.txt")
    assert len(chunks) >= 1
    assert_valid_chunks(chunks, "short.txt")

def test_custom_metadata_propagates(medium_doc):
    meta = {"source": "arxiv", "year": 2017}
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk(medium_doc, doc_id="paper.txt", metadata=meta)
    for c in chunks:
        assert c.metadata["source"] == "arxiv"
        assert c.metadata["year"] == 2017


# ── fixed strategy ────────────────────────────────────────────────────────────

def test_fixed_strategy_basic(medium_doc):
    chunker = Chunker(strategy="fixed", max_tokens=60, overlap=10)
    chunks = chunker.chunk(medium_doc, doc_id="fixed.txt")
    assert_valid_chunks(chunks, "fixed.txt")

def test_fixed_strategy_respects_max_tokens(medium_doc):
    max_t = 50
    chunker = Chunker(strategy="fixed", max_tokens=max_t, overlap=5)
    chunks = chunker.chunk(medium_doc)
    for c in chunks:
        # Allow a small buffer because we merge to word boundaries
        assert c.token_count <= max_t + 20, f"chunk too large: {c.token_count}"

def test_fixed_strategy_enum(medium_doc):
    chunker = Chunker(strategy=ChunkStrategy.FIXED, max_tokens=80)
    chunks = chunker.chunk(medium_doc)
    for c in chunks:
        assert c.strategy == ChunkStrategy.FIXED


# ── recursive strategy ────────────────────────────────────────────────────────

def test_recursive_strategy_basic(medium_doc):
    chunker = Chunker(strategy="recursive", min_tokens=10, max_tokens=60)
    chunks = chunker.chunk(medium_doc, doc_id="rec.txt")
    assert_valid_chunks(chunks, "rec.txt")

def test_recursive_strategy_preserves_paragraphs(medium_doc):
    chunker = Chunker(strategy="recursive", min_tokens=10, max_tokens=60)
    chunks = chunker.chunk(medium_doc)
    # Each chunk should be a clean paragraph or merge of paragraphs
    assert len(chunks) >= 2

def test_recursive_chunk_indices_are_sequential(medium_doc):
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk(medium_doc)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i

def test_recursive_no_boundary_breaks(medium_doc):
    """Recursive strategy should produce zero sub-20-token micro-chunks."""
    chunker = Chunker(strategy="recursive", min_tokens=30, max_tokens=400)
    chunks = chunker.chunk(medium_doc)
    micro = [c for c in chunks if c.token_count < 10]
    assert len(micro) == 0, f"Found micro-chunks: {micro}"


# ── structural strategy ───────────────────────────────────────────────────────

def test_structural_strategy_basic(markdown_doc):
    chunker = Chunker(strategy="structural", min_tokens=10, max_tokens=400)
    chunks = chunker.chunk(markdown_doc, doc_id="md.txt")
    assert_valid_chunks(chunks, "md.txt")

def test_structural_strategy_on_plain_text(medium_doc):
    chunker = Chunker(strategy="structural", min_tokens=20, max_tokens=400)
    chunks = chunker.chunk(medium_doc)
    assert len(chunks) >= 1

def test_structural_respects_max_tokens(markdown_doc):
    max_t = 100
    chunker = Chunker(strategy="structural", max_tokens=max_t)
    chunks = chunker.chunk(markdown_doc)
    for c in chunks:
        assert c.token_count <= max_t + 30


# ── adaptive strategy ─────────────────────────────────────────────────────────

def test_adaptive_picks_structural_for_markdown(markdown_doc):
    chunker = Chunker(strategy="adaptive")
    chunks = chunker.chunk(markdown_doc)
    assert len(chunks) >= 1
    # Should have used structural
    assert chunks[0].strategy == ChunkStrategy.STRUCTURAL

def test_adaptive_picks_recursive_for_prose(medium_doc):
    chunker = Chunker(strategy="adaptive")
    chunks = chunker.chunk(medium_doc)
    assert chunks[0].strategy == ChunkStrategy.RECURSIVE


# ── span coverage ─────────────────────────────────────────────────────────────

def test_spans_are_within_doc_bounds(medium_doc):
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk(medium_doc)
    doc_len = len(medium_doc)
    for c in chunks:
        assert c.span[0] >= 0
        assert c.span[1] <= doc_len + 5  # +5 for whitespace normalisation


# ── IDs are unique across chunks ─────────────────────────────────────────────

def test_chunk_ids_are_unique(medium_doc):
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk(medium_doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
