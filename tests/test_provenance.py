"""Tests for ProvenanceStore."""
import pytest
from chunkops import Chunker, ProvenanceStore


@pytest.fixture
def store_with_chunks(medium_doc):
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk(medium_doc, doc_id="paper.txt")
    store = ProvenanceStore()
    store.register(chunks)
    return store, chunks


def test_register_and_count(store_with_chunks):
    store, chunks = store_with_chunks
    assert store.count() == len(chunks)


def test_trace_returns_correct_chunk(store_with_chunks):
    store, chunks = store_with_chunks
    target = chunks[0]
    retrieved = store.trace(target.id)
    assert retrieved.id == target.id
    assert retrieved.doc_id == target.doc_id
    assert retrieved.token_count == target.token_count
    assert retrieved.text == target.text


def test_trace_preserves_span(store_with_chunks):
    store, chunks = store_with_chunks
    for chunk in chunks:
        retrieved = store.trace(chunk.id)
        assert retrieved.span == chunk.span


def test_trace_nonexistent_raises_keyerror(store_with_chunks):
    store, _ = store_with_chunks
    with pytest.raises(KeyError):
        store.trace("nonexistent_id")


def test_trace_doc_returns_all_chunks(store_with_chunks, medium_doc):
    store, chunks = store_with_chunks
    doc_chunks = store.trace_doc("paper.txt")
    assert len(doc_chunks) == len(chunks)
    assert [c.chunk_index for c in doc_chunks] == list(range(len(chunks)))


def test_trace_doc_unknown_returns_empty(store_with_chunks):
    store, _ = store_with_chunks
    result = store.trace_doc("unknown_doc.txt")
    assert result == []


def test_docs_list(store_with_chunks, medium_doc):
    store, _ = store_with_chunks
    docs = store.docs()
    assert "paper.txt" in docs


def test_register_is_idempotent(store_with_chunks):
    store, chunks = store_with_chunks
    count_before = store.count()
    store.register(chunks)  # register same chunks again
    assert store.count() == count_before  # INSERT OR IGNORE


def test_clear(store_with_chunks):
    store, _ = store_with_chunks
    store.clear()
    assert store.count() == 0


def test_persistent_store(tmp_path, medium_doc):
    db_path = tmp_path / "prov.db"
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk(medium_doc, doc_id="persist.txt")

    # Write
    store = ProvenanceStore(db_path)
    store.register(chunks)
    store.close()

    # Read back in new instance
    store2 = ProvenanceStore(db_path)
    assert store2.count() == len(chunks)
    retrieved = store2.trace(chunks[0].id)
    assert retrieved.text == chunks[0].text
    store2.close()


def test_multiple_docs(medium_doc, markdown_doc):
    chunker = Chunker(strategy="recursive")
    c1 = chunker.chunk(medium_doc, doc_id="doc1.txt")
    c2 = chunker.chunk(markdown_doc, doc_id="doc2.txt")

    store = ProvenanceStore()
    store.register(c1)
    store.register(c2)

    assert store.count() == len(c1) + len(c2)
    assert set(store.docs()) == {"doc1.txt", "doc2.txt"}
    assert len(store.trace_doc("doc1.txt")) == len(c1)
    assert len(store.trace_doc("doc2.txt")) == len(c2)
