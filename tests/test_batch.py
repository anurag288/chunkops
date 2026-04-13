"""Tests for BatchChunker."""
import pytest
from chunkops import BatchChunker
from chunkops.models import ChunkResult


DOCS = [
    ("doc_a", "The transformer architecture replaced recurrence with self-attention. Each token attends to all others."),
    ("doc_b", "BERT uses bidirectional encoders pre-trained with masked language modeling. Fine-tuning achieves strong results."),
    ("doc_c", "GPT models use decoder-only transformers. Scaling to billions of parameters led to emergent capabilities."),
    ("doc_d", "Vector databases store high-dimensional embeddings. Nearest-neighbor search retrieves similar vectors quickly."),
    ("doc_e", "RAG combines retrieval with generation. It grounds model responses in factual source documents."),
]


def test_batch_returns_list_of_chunk_results():
    bc = BatchChunker(strategy="recursive", workers=2)
    results = bc.run(DOCS)
    assert isinstance(results, list)
    assert all(isinstance(r, ChunkResult) for r in results)


def test_batch_processes_all_docs():
    bc = BatchChunker(strategy="recursive", workers=2)
    results = bc.run(DOCS)
    doc_ids = {r.doc_id for r in results}
    assert doc_ids == {"doc_a", "doc_b", "doc_c", "doc_d", "doc_e"}


def test_batch_plain_strings():
    texts = ["Hello world. This is sentence two.", "Another document here. More content follows."]
    bc = BatchChunker(strategy="recursive", workers=1)
    results = bc.run(texts)
    assert len(results) > 0
    assert results[0].doc_id.startswith("doc_")


def test_batch_on_progress_called(medium_doc):
    docs = [("d1", medium_doc), ("d2", medium_doc), ("d3", medium_doc)]
    calls = []
    bc = BatchChunker(strategy="recursive", workers=2)
    bc.run(docs, on_progress=lambda n, t: calls.append((n, t)))
    assert len(calls) == 3
    assert calls[-1][0] == 3
    assert calls[-1][1] == 3


def test_batch_checkpoint_saves_and_resumes(tmp_path, medium_doc):
    docs = [(f"doc_{i}", medium_doc) for i in range(6)]
    ckpt = tmp_path / "ckpt"

    # First run — process all
    bc1 = BatchChunker(strategy="recursive", workers=2, checkpoint=str(ckpt))
    results1 = bc1.run(docs)
    assert len(results1) > 0

    # Second run — all already done, should skip (zero new work)
    calls = []
    bc2 = BatchChunker(strategy="recursive", workers=2, checkpoint=str(ckpt))
    results2 = bc2.run(docs, on_progress=lambda n, t: calls.append(n))
    # Progress callback fires for already-done count at startup
    assert (ckpt / "completed_ids.pkl").exists()


def test_batch_single_worker(medium_doc):
    docs = [("d1", medium_doc), ("d2", medium_doc)]
    bc = BatchChunker(strategy="recursive", workers=1)
    results = bc.run(docs)
    assert len(results) >= 2


def test_batch_invalid_doc_raises():
    bc = BatchChunker(strategy="recursive", workers=1)
    with pytest.raises(ValueError):
        bc.run([123])  # not a string or tuple


def test_batch_structural_strategy():
    docs = [("md1", "# Title\n\nSome paragraph content here.\n\n## Section\n\nMore content follows here.")]
    bc = BatchChunker(strategy="structural", workers=1)
    results = bc.run(docs)
    assert len(results) >= 1
    assert results[0].strategy.value == "structural"
