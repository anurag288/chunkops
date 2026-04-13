"""
examples/basic_usage.py
-----------------------
Demonstrates the core chunkops workflow:
  1. Chunk a document with different strategies
  2. Inspect ChunkResult provenance fields
  3. Store and trace chunks via ProvenanceStore
  4. Run a benchmark across strategies
  5. Batch-process multiple documents

Run:
    python examples/basic_usage.py
"""

from chunkops import Chunker, ProvenanceStore, BatchChunker, benchmark

DOC = """
The transformer architecture, introduced in 'Attention is All You Need' (2017),
replaced recurrence with self-attention. Each token attends to all others in the
sequence, enabling parallel computation and capturing long-range dependencies.

BERT uses bidirectional encoders pre-trained with masked language modeling.
It learns deep context from both left and right of each token. Fine-tuning BERT
on downstream tasks like classification achieved state-of-the-art results in 2018.

GPT models use decoder-only transformers trained autoregressively. They predict
the next token given all previous tokens. Scaling GPT to billions of parameters
led to emergent capabilities like few-shot and zero-shot generalization.

Vector databases store high-dimensional embeddings produced by models like these.
Approximate nearest-neighbor search retrieves the k most similar vectors to a
query in milliseconds. Pinecone, Qdrant, and pgvector are common choices.
""".strip()


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Basic chunking ─────────────────────────────────────────────────────────

section("1. Recursive chunking (recommended default)")

# Use tighter token limits so the demo doc splits into multiple chunks
chunker = Chunker(strategy="recursive", min_tokens=30, max_tokens=80)
chunks = chunker.chunk(DOC, doc_id="transformers_intro.txt")

print(f"Produced {len(chunks)} chunks:\n")
for c in chunks:
    preview = c.text[:70].replace("\n", " ")
    print(f"  [{c.id}]  tokens={c.token_count:>3}  span={c.span}  {preview!r}")


# ── 2. Inspect provenance fields ──────────────────────────────────────────────

section("2. Provenance fields on a single ChunkResult")

# Use the first chunk (always exists); show second if available
c = chunks[min(1, len(chunks) - 1)]
print(f"  id:           {c.id}")
print(f"  doc_id:       {c.doc_id}")
print(f"  chunk_index:  {c.chunk_index}")
print(f"  span:         {c.span}  (char offsets into original text)")
print(f"  token_count:  {c.token_count}")
print(f"  strategy:     {c.strategy.value}")
print(f"  merged_from:  {c.merged_from}  (sentence indices merged into this chunk)")
print(f"\n  text:\n    {c.text[:120]!r}")


# ── 3. ProvenanceStore ────────────────────────────────────────────────────────

section("3. ProvenanceStore — register and trace back")

store = ProvenanceStore()          # in-memory
store.register(chunks)
print(f"  Store contains {store.count()} chunks across {store.docs()} docs\n")

target_id = chunks[0].id
origin = store.trace(target_id)
print(f"  store.trace({target_id!r})")
print(f"    → doc_id:   {origin.doc_id}")
print(f"    → span:     {origin.span}")
print(f"    → tokens:   {origin.token_count}")
print(f"    → strategy: {origin.strategy.value}")
print(f"    → text:     {origin.text[:80]!r}")


# ── 4. Strategy comparison ────────────────────────────────────────────────────

section("4. Compare strategies side-by-side")

for strategy in ["fixed", "recursive", "structural"]:
    c2 = Chunker(strategy=strategy, min_tokens=20, max_tokens=80)
    ch = c2.chunk(DOC)
    tokens = [x.token_count for x in ch]
    avg = sum(tokens) / len(tokens) if tokens else 0
    breaks = sum(1 for t in tokens if t < 20)
    print(f"  {strategy:<12}  chunks={len(ch):>2}  avg_tokens={avg:>5.1f}  micro_chunks={breaks}")


# ── 5. Benchmark ──────────────────────────────────────────────────────────────

section("5. Benchmark — find best strategy for your corpus")

result = benchmark(
    docs=[DOC],
    strategies=["fixed", "recursive", "structural"],
    metric="coherence",
)
result.report()
print(f"  Recommended strategy for this corpus: {result.best_strategy}")


# ── 6. BatchChunker ───────────────────────────────────────────────────────────

section("6. BatchChunker — process many docs with progress + checkpoint")

corpus = [
    ("doc_transformers", DOC),
    ("doc_bert", "BERT is a transformer model trained with masked language modeling. It achieves strong results on many NLP tasks across benchmarks."),
    ("doc_gpt", "GPT uses autoregressive training to predict the next token. Scaling laws show consistent improvement with more compute and data."),
    ("doc_rag", "RAG combines a retriever with a language model generator. It grounds responses in retrieved document chunks to reduce hallucinations."),
]

bc = BatchChunker(strategy="recursive", workers=2, min_tokens=20, max_tokens=80)
all_chunks = bc.run(
    corpus,
    on_progress=lambda n, t: print(f"  progress: {n}/{t} docs"),
)

print(f"\n  Total chunks produced: {len(all_chunks)}")
by_doc = {}
for ch in all_chunks:
    by_doc.setdefault(ch.doc_id, []).append(ch)
for doc_id, doc_chunks in sorted(by_doc.items()):
    print(f"  {doc_id}: {len(doc_chunks)} chunk(s)")

print("\nDone. All examples completed successfully.")
