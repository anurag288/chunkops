"""Tests for ChunkResult and ChunkStrategy models."""
import pytest
from chunkops.models import ChunkResult, ChunkStrategy


def test_chunk_result_id_is_stable():
    c1 = ChunkResult(text="hello world", doc_id="doc.txt", chunk_index=0,
                     span=(0, 11), token_count=2, strategy=ChunkStrategy.RECURSIVE)
    c2 = ChunkResult(text="hello world", doc_id="doc.txt", chunk_index=0,
                     span=(0, 11), token_count=2, strategy=ChunkStrategy.RECURSIVE)
    assert c1.id == c2.id, "Same content+position must produce same ID"


def test_chunk_result_id_differs_by_index():
    c1 = ChunkResult(text="hello world", doc_id="doc.txt", chunk_index=0,
                     span=(0, 11), token_count=2, strategy=ChunkStrategy.RECURSIVE)
    c2 = ChunkResult(text="hello world", doc_id="doc.txt", chunk_index=1,
                     span=(0, 11), token_count=2, strategy=ChunkStrategy.RECURSIVE)
    assert c1.id != c2.id


def test_chunk_result_repr():
    c = ChunkResult(text="The quick brown fox", doc_id="test.txt", chunk_index=0,
                    span=(0, 19), token_count=4, strategy=ChunkStrategy.FIXED)
    r = repr(c)
    assert "ChunkResult" in r
    assert c.id in r


def test_chunk_strategy_enum():
    assert ChunkStrategy("recursive") == ChunkStrategy.RECURSIVE
    assert ChunkStrategy("semantic") == ChunkStrategy.SEMANTIC
    with pytest.raises(ValueError):
        ChunkStrategy("nonexistent")


def test_chunk_result_metadata_default_empty():
    c = ChunkResult(text="test", doc_id="d", chunk_index=0,
                    span=(0, 4), token_count=1, strategy=ChunkStrategy.STRUCTURAL)
    assert c.metadata == {}
    assert c.merged_from == []


def test_chunk_result_with_metadata():
    meta = {"source": "pdf", "page": 3}
    c = ChunkResult(text="test", doc_id="d", chunk_index=0,
                    span=(0, 4), token_count=1, strategy=ChunkStrategy.RECURSIVE,
                    metadata=meta)
    assert c.metadata["source"] == "pdf"
    assert c.metadata["page"] == 3
