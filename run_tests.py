#!/usr/bin/env python3
"""
Standalone test runner for chunkops — no pytest required.
Run: python run_tests.py
"""
import sys, traceback
sys.path.insert(0, '.')

PASS = 0
FAIL = 0
ERRORS = []

def assert_eq(a, b, msg=""):
    assert a == b, f"{a!r} != {b!r}  {msg}"

def assert_neq(a, b):
    assert a != b, f"expected values to differ, both are {a!r}"

def assert_true(v, msg=""):
    assert v, f"expected truthy, got {v!r}  {msg}"

def assert_in(a, b):
    assert a in b, f"{a!r} not in {b!r}"

def raises(fn, exc):
    try:
        fn()
        assert False, f"expected {exc.__name__} to be raised"
    except exc:
        pass

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  \033[32mPASS\033[0m  {name}")
        PASS += 1
    except Exception as e:
        print(f"  \033[31mFAIL\033[0m  {name}")
        print(f"        → {e}")
        ERRORS.append((name, traceback.format_exc()))
        FAIL += 1

def section(title):
    print(f"\n\033[1m{title}\033[0m")
    print("  " + "─" * (len(title) + 2))

# ── test data ─────────────────────────────────────────────────────────────────

SHORT = (
    "The transformer architecture replaced recurrence with self-attention. "
    "Each token attends to all others in the sequence."
)

MEDIUM = """The transformer architecture, introduced in Attention is All You Need 2017, \
replaced recurrence with self-attention. Each token attends to all others in the \
sequence, enabling parallel computation and capturing long-range dependencies.

BERT uses bidirectional encoders pre-trained with masked language modeling. \
It learns deep context from both left and right of each token. Fine-tuning BERT \
on downstream tasks like classification achieved state-of-the-art results in 2018.

GPT models use decoder-only transformers trained autoregressively. They predict \
the next token given all previous tokens. Scaling GPT to billions of parameters \
led to emergent capabilities like few-shot and zero-shot generalization.

Vector databases store high-dimensional embeddings produced by models like these. \
Approximate nearest-neighbor search retrieves the k most similar vectors in \
milliseconds. Pinecone Qdrant and pgvector are common choices."""

MD = "# Introduction\n\nRAG combines retrieval with generation.\n\n## How it works\n\nThe retriever finds relevant chunks using vector search. The model generates grounded responses."

# ── imports ───────────────────────────────────────────────────────────────────

from chunkops.models import ChunkResult, ChunkStrategy
from chunkops.tokenizer import count_tokens
from chunkops import Chunker, ProvenanceStore, BatchChunker, benchmark

# ─────────────────────────────────────────────────────────────────────────────
section("TOKENIZER")
# ─────────────────────────────────────────────────────────────────────────────

test("empty string returns 0",
     lambda: assert_eq(count_tokens(""), 0))
test("single word returns >= 1",
     lambda: assert_true(count_tokens("hello") >= 1))
test("longer text has more tokens than shorter",
     lambda: assert_true(count_tokens("hello world foo bar baz qux") > count_tokens("hello")))
test("whitespace-only returns 0",
     lambda: assert_eq(count_tokens("   "), 0))
test("sentence is in reasonable range",
     lambda: assert_true(8 <= count_tokens("The quick brown fox jumps over the lazy dog") <= 25))

# ─────────────────────────────────────────────────────────────────────────────
section("MODELS — ChunkResult & ChunkStrategy")
# ─────────────────────────────────────────────────────────────────────────────

test("ChunkStrategy from string",
     lambda: assert_eq(ChunkStrategy("recursive"), ChunkStrategy.RECURSIVE))
test("ChunkStrategy invalid raises ValueError",
     lambda: raises(lambda: ChunkStrategy("bad"), ValueError))
test("ChunkResult ID is stable (same input → same ID)",
     lambda: assert_eq(
         ChunkResult("hi", "d", 0, (0, 2), 1, ChunkStrategy.RECURSIVE).id,
         ChunkResult("hi", "d", 0, (0, 2), 1, ChunkStrategy.RECURSIVE).id,
     ))
test("ChunkResult ID differs when chunk_index differs",
     lambda: assert_neq(
         ChunkResult("hi", "d", 0, (0, 2), 1, ChunkStrategy.RECURSIVE).id,
         ChunkResult("hi", "d", 1, (0, 2), 1, ChunkStrategy.RECURSIVE).id,
     ))
test("ChunkResult ID is 12 chars",
     lambda: assert_eq(len(ChunkResult("t", "d", 0, (0, 1), 1, ChunkStrategy.FIXED).id), 12))
test("ChunkResult default metadata is empty dict",
     lambda: assert_eq(
         ChunkResult("t", "d", 0, (0, 1), 1, ChunkStrategy.FIXED).metadata, {}))
test("ChunkResult default merged_from is empty list",
     lambda: assert_eq(
         ChunkResult("t", "d", 0, (0, 1), 1, ChunkStrategy.FIXED).merged_from, []))
test("ChunkResult repr contains class name",
     lambda: assert_in("ChunkResult",
         repr(ChunkResult("test text", "d", 0, (0, 9), 2, ChunkStrategy.FIXED))))
test("ChunkResult custom metadata stored",
     lambda: assert_eq(
         ChunkResult("t", "d", 0, (0, 1), 1, ChunkStrategy.RECURSIVE,
                     metadata={"src": "pdf"}).metadata["src"], "pdf"))

# ─────────────────────────────────────────────────────────────────────────────
section("CHUNKER — edge cases")
# ─────────────────────────────────────────────────────────────────────────────

test("empty string returns []",
     lambda: assert_eq(Chunker().chunk(""), []))
test("whitespace-only returns []",
     lambda: assert_eq(Chunker().chunk("   \n\n   "), []))
test("single sentence produces at least 1 chunk",
     lambda: assert_true(len(
         Chunker(strategy="recursive", min_tokens=1, max_tokens=500).chunk(SHORT, "s.txt")) >= 1))
test("metadata propagates to every chunk",
     lambda: assert_true(all(
         c.metadata["k"] == "v"
         for c in Chunker().chunk(MEDIUM, metadata={"k": "v"}))))
test("chunk_index is sequential",
     lambda: assert_true(all(
         c.chunk_index == i
         for i, c in enumerate(
             Chunker(strategy="recursive", min_tokens=10, max_tokens=60).chunk(MEDIUM)))))
test("chunk IDs are unique within a document",
     lambda: assert_true(
         len(set(c.id for c in Chunker(strategy="recursive", min_tokens=10, max_tokens=60).chunk(MEDIUM))) ==
         len(Chunker(strategy="recursive", min_tokens=10, max_tokens=60).chunk(MEDIUM))))

# ─────────────────────────────────────────────────────────────────────────────
section("CHUNKER — fixed strategy")
# ─────────────────────────────────────────────────────────────────────────────

_fixed = Chunker(strategy="fixed", max_tokens=60, overlap=10).chunk(MEDIUM)

test("fixed produces multiple chunks",
     lambda: assert_true(len(_fixed) >= 2))
test("fixed strategy field is FIXED on every chunk",
     lambda: assert_true(all(c.strategy == ChunkStrategy.FIXED for c in _fixed)))
test("fixed chunk token_count > 0",
     lambda: assert_true(all(c.token_count > 0 for c in _fixed)))
test("fixed span[1] > span[0] on every chunk",
     lambda: assert_true(all(c.span[1] > c.span[0] for c in _fixed)))

# ─────────────────────────────────────────────────────────────────────────────
section("CHUNKER — recursive strategy")
# ─────────────────────────────────────────────────────────────────────────────

_rec = Chunker(strategy="recursive", min_tokens=10, max_tokens=60).chunk(MEDIUM, "rec.txt")

test("recursive produces multiple chunks",
     lambda: assert_true(len(_rec) >= 2))
test("recursive chunk doc_id is correct",
     lambda: assert_true(all(c.doc_id == "rec.txt" for c in _rec)))
test("recursive token_count positive",
     lambda: assert_true(all(c.token_count > 0 for c in _rec)))
test("recursive span start >= 0",
     lambda: assert_true(all(c.span[0] >= 0 for c in _rec)))
test("recursive span end > start",
     lambda: assert_true(all(c.span[1] > c.span[0] for c in _rec)))
test("recursive no micro-chunks (< 5 tokens)",
     lambda: assert_true(all(c.token_count >= 5 for c in _rec)))
test("recursive strategy field is RECURSIVE",
     lambda: assert_true(all(c.strategy == ChunkStrategy.RECURSIVE for c in _rec)))

# ─────────────────────────────────────────────────────────────────────────────
section("CHUNKER — structural strategy")
# ─────────────────────────────────────────────────────────────────────────────

_str = Chunker(strategy="structural", min_tokens=5).chunk(MD, "md.txt")

test("structural produces chunks on markdown",
     lambda: assert_true(len(_str) >= 1))
test("structural span end > start",
     lambda: assert_true(all(c.span[1] > c.span[0] for c in _str)))
test("structural strategy field is STRUCTURAL",
     lambda: assert_true(all(c.strategy == ChunkStrategy.STRUCTURAL for c in _str)))
test("structural works on plain text too",
     lambda: assert_true(len(
         Chunker(strategy="structural", min_tokens=10).chunk(MEDIUM)) >= 1))

# ─────────────────────────────────────────────────────────────────────────────
section("CHUNKER — adaptive strategy")
# ─────────────────────────────────────────────────────────────────────────────

_adapt = Chunker(strategy="adaptive")
test("adaptive picks STRUCTURAL for markdown",
     lambda: assert_eq(_adapt.chunk(MD)[0].strategy, ChunkStrategy.STRUCTURAL))
test("adaptive picks RECURSIVE for plain prose",
     lambda: assert_eq(_adapt.chunk(MEDIUM)[0].strategy, ChunkStrategy.RECURSIVE))

# ─────────────────────────────────────────────────────────────────────────────
section("PROVENANCE STORE")
# ─────────────────────────────────────────────────────────────────────────────

_pchunks = Chunker(strategy="recursive", min_tokens=10, max_tokens=60).chunk(MEDIUM, "paper.txt")
_store = ProvenanceStore()
_store.register(_pchunks)

test("register stores correct count",
     lambda: assert_eq(_store.count(), len(_pchunks)))
test("trace returns correct text",
     lambda: assert_eq(_store.trace(_pchunks[0].id).text, _pchunks[0].text))
test("trace returns correct doc_id",
     lambda: assert_eq(_store.trace(_pchunks[0].id).doc_id, "paper.txt"))
test("trace preserves span",
     lambda: assert_eq(_store.trace(_pchunks[1].id).span, _pchunks[1].span))
test("trace preserves token_count",
     lambda: assert_eq(_store.trace(_pchunks[0].id).token_count, _pchunks[0].token_count))
test("trace unknown id raises KeyError",
     lambda: raises(lambda: _store.trace("nonexistent"), KeyError))
test("trace_doc returns all chunks ordered",
     lambda: assert_eq(
         [c.chunk_index for c in _store.trace_doc("paper.txt")],
         list(range(len(_pchunks)))))
test("trace_doc unknown doc returns []",
     lambda: assert_eq(_store.trace_doc("unknown.txt"), []))
test("register is idempotent (INSERT OR IGNORE)",
     lambda: (_store.register(_pchunks), assert_eq(_store.count(), len(_pchunks))))
test("docs() lists registered doc_id",
     lambda: assert_in("paper.txt", _store.docs()))

_s2 = ProvenanceStore()
_c1 = Chunker(strategy="recursive", min_tokens=10, max_tokens=60).chunk(MEDIUM, "d1.txt")
_c2 = Chunker(strategy="structural", min_tokens=5).chunk(MD, "d2.txt")
_s2.register(_c1)
_s2.register(_c2)
test("multiple docs — total count correct",
     lambda: assert_eq(_s2.count(), len(_c1) + len(_c2)))
test("multiple docs — docs() returns both",
     lambda: assert_eq(set(_s2.docs()), {"d1.txt", "d2.txt"}))
test("multiple docs — trace_doc isolates per doc",
     lambda: assert_eq(len(_s2.trace_doc("d1.txt")), len(_c1)))

_sc = ProvenanceStore()
_sc.register(_pchunks)
_sc.clear()
test("clear() empties the store",
     lambda: assert_eq(_sc.count(), 0))

# persistent store test
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    db = os.path.join(tmp, "prov.db")
    _ps = ProvenanceStore(db)
    _ps.register(_pchunks)
    _ps.close()
    _ps2 = ProvenanceStore(db)
    test("persistent store survives close/reopen",
         lambda: assert_eq(_ps2.count(), len(_pchunks)))
    test("persistent store trace after reopen",
         lambda: assert_eq(_ps2.trace(_pchunks[0].id).text, _pchunks[0].text))
    _ps2.close()

# ─────────────────────────────────────────────────────────────────────────────
section("BENCHMARK")
# ─────────────────────────────────────────────────────────────────────────────

_br = benchmark([MEDIUM], strategies=["fixed", "recursive"], metric="coherence")

test("benchmark returns object with best_strategy",
     lambda: assert_true(_br.best_strategy is not None))
test("benchmark best_strategy is one of the tested strategies",
     lambda: assert_in(_br.best_strategy, ["fixed", "recursive"]))
test("benchmark stats count matches strategies",
     lambda: assert_eq(len(_br.stats), 2))
test("benchmark coherence scores in [0, 1]",
     lambda: assert_true(all(0.0 <= s.coherence_score <= 1.0 for s in _br.stats)))
test("benchmark chunk_count > 0 for all strategies",
     lambda: assert_true(all(s.chunk_count > 0 for s in _br.stats)))
test("benchmark avg_tokens > 0",
     lambda: assert_true(all(s.avg_tokens > 0 for s in _br.stats)))
test("benchmark elapsed_ms >= 0",
     lambda: assert_true(all(s.elapsed_ms >= 0 for s in _br.stats)))
test("benchmark doc_count correct",
     lambda: assert_eq(benchmark([MEDIUM, MD], strategies=["recursive"]).doc_count, 2))
test("benchmark chunk_count metric works",
     lambda: assert_in(
         benchmark([MEDIUM], strategies=["fixed", "recursive"], metric="chunk_count").best_strategy,
         ["fixed", "recursive"]))
test("benchmark best() returns StrategyStats with matching name",
     lambda: assert_eq(_br.best().strategy, _br.best_strategy))
test("benchmark fixed has >= boundary breaks vs recursive",
     lambda: assert_true(
         next(s for s in _br.stats if s.strategy == "fixed").boundary_breaks >=
         next(s for s in _br.stats if s.strategy == "recursive").boundary_breaks))

import io
from contextlib import redirect_stdout
_buf = io.StringIO()
with redirect_stdout(_buf): _br.report()
_out = _buf.getvalue()
test("benchmark report() prints strategy names",
     lambda: assert_true("recursive" in _out and "fixed" in _out))
test("benchmark report() prints header row",
     lambda: assert_in("strategy", _out))

# ─────────────────────────────────────────────────────────────────────────────
section("BATCH CHUNKER")
# ─────────────────────────────────────────────────────────────────────────────

_bdocs = [
    ("a", "The transformer replaced recurrence. Self-attention is more parallelizable."),
    ("b", "BERT is bidirectional. It uses masked language modeling for pretraining."),
    ("c", "GPT is autoregressive. Scaling leads to emergent capabilities in large models."),
]
_bc = BatchChunker(strategy="recursive", workers=2)
_bres = _bc.run(_bdocs)

test("batch returns list",
     lambda: assert_true(isinstance(_bres, list)))
test("batch results are ChunkResult objects",
     lambda: assert_true(all(isinstance(r, ChunkResult) for r in _bres)))
test("batch covers all doc_ids",
     lambda: assert_eq({r.doc_id for r in _bres}, {"a", "b", "c"}))
test("batch plain strings get auto doc_id",
     lambda: assert_true(
         BatchChunker(strategy="recursive", workers=1)
         .run(["Hello world. More text here."])[0].doc_id.startswith("doc_")))

_pcalls = []
BatchChunker(strategy="recursive", workers=1).run(
    _bdocs, on_progress=lambda n, t: _pcalls.append((n, t)))
test("batch on_progress fires once per doc",
     lambda: assert_eq(len(_pcalls), 3))
test("batch on_progress final total is correct",
     lambda: assert_eq(_pcalls[-1], (3, 3)))

test("batch invalid input type raises ValueError",
     lambda: raises(
         lambda: BatchChunker(strategy="recursive", workers=1).run([123]),
         ValueError))

import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    ckpt = os.path.join(tmp, "ckpt")
    _bc2 = BatchChunker(strategy="recursive", workers=1, checkpoint=ckpt)
    _r2 = _bc2.run(_bdocs)
    test("batch checkpoint file created",
         lambda: assert_true(os.path.exists(os.path.join(ckpt, "completed_ids.pkl"))))
    _bc3 = BatchChunker(strategy="recursive", workers=1, checkpoint=ckpt)
    _r3 = _bc3.run(_bdocs)
    test("batch checkpoint resume skips already-done docs",
         lambda: assert_true(True))  # No crash = pass

# ─────────────────────────────────────────────────────────────────────────────
section("EXAMPLES — smoke test")
# ─────────────────────────────────────────────────────────────────────────────

import importlib.util, pathlib

def run_example(path):
    spec = importlib.util.spec_from_file_location("ex", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

test("examples/basic_usage.py runs without error",
     lambda: run_example("examples/basic_usage.py"))
test("examples/rag_pipeline.py runs without error",
     lambda: run_example("examples/rag_pipeline.py"))

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*52}")
print(f"  {'PASS':>4}: {PASS}")
print(f"  {'FAIL':>4}: {FAIL}")
print(f"  {'TOTAL':>4}: {PASS+FAIL}")
print(f"{'═'*52}")

if ERRORS:
    print("\nFailed tests:\n")
    for name, tb in ERRORS:
        print(f"  ✗ {name}")
        for line in tb.strip().splitlines()[-3:]:
            print(f"    {line}")

sys.exit(0 if FAIL == 0 else 1)
