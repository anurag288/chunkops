"""
examples/rag_pipeline.py
------------------------
Simulates a complete RAG pipeline using chunkops:

  load docs → chunk → store provenance → fake embed → fake retrieve
            → answer query → trace answer back to source passage

No external API key required — embeddings are mocked with random vectors
so you can run this immediately without any paid service.

Run:
    python examples/rag_pipeline.py
"""

from __future__ import annotations

import random
import math
from typing import Dict, List, Tuple

from chunkops import Chunker, ProvenanceStore
from chunkops.models import ChunkResult

random.seed(42)

# ── Fake vector store (no real embeddings needed to run this demo) ────────────

def fake_embed(text: str, dim: int = 64) -> List[float]:
    """Deterministic fake embedding based on text hash."""
    h = hash(text) % (2**32)
    rng = random.Random(h)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x**2 for x in vec))
    return [x / norm for x in vec]


def cosine_sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    def __init__(self):
        self._store: List[Tuple[ChunkResult, List[float]]] = []

    def add(self, chunks: List[ChunkResult]):
        for c in chunks:
            self._store.append((c, fake_embed(c.text)))

    def search(self, query: str, k: int = 3) -> List[ChunkResult]:
        qv = fake_embed(query)
        scored = [(cosine_sim(qv, vec), chunk) for chunk, vec in self._store]
        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:k]]


# ── Sample knowledge base ─────────────────────────────────────────────────────

DOCS: Dict[str, str] = {
    "transformers.txt": """
The transformer architecture, introduced in 'Attention is All You Need' (2017),
replaced recurrence with self-attention. Each token attends to all others in the
sequence, enabling parallel computation and capturing long-range dependencies.
The architecture consists of an encoder and decoder, each built from stacked
multi-head attention layers and feed-forward networks.
    """.strip(),

    "bert.txt": """
BERT (Bidirectional Encoder Representations from Transformers) uses bidirectional
encoders pre-trained with masked language modeling. It learns deep context from
both left and right of each token simultaneously. Fine-tuning BERT on downstream
tasks like classification, NER, and question answering achieved state-of-the-art
results across the GLUE benchmark in 2018.
    """.strip(),

    "gpt.txt": """
GPT models use decoder-only transformers trained autoregressively. They predict
the next token given all previous tokens. Scaling GPT to billions of parameters
led to emergent capabilities like few-shot and zero-shot generalization.
GPT-3 demonstrated that large-scale pretraining enables strong in-context learning
without any gradient updates at inference time.
    """.strip(),

    "rag.txt": """
Retrieval-Augmented Generation (RAG) combines a retrieval system with a language
model generator. The retriever finds relevant document chunks using vector
similarity search. These chunks are passed as context to the language model,
which generates a response grounded in the retrieved content. RAG reduces
hallucinations by anchoring the model to factual source documents.
    """.strip(),
}


def section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


# ── Step 1: Ingest and chunk all documents ────────────────────────────────────

section("Step 1 — chunk knowledge base")

chunker = Chunker(strategy="recursive", min_tokens=40, max_tokens=200)
store = ProvenanceStore()
vector_store = VectorStore()
all_chunks: List[ChunkResult] = []

for doc_id, text in DOCS.items():
    chunks = chunker.chunk(text, doc_id=doc_id)
    all_chunks.extend(chunks)
    store.register(chunks)
    vector_store.add(chunks)
    print(f"  {doc_id}: {len(chunks)} chunk(s), "
          f"tokens: {[c.token_count for c in chunks]}")

print(f"\n  Total chunks indexed: {len(all_chunks)}")
print(f"  ProvenanceStore: {store}")


# ── Step 2: Simulate RAG queries ──────────────────────────────────────────────

QUERIES = [
    "How does BERT use context from both directions?",
    "What made GPT-3 capable of few-shot learning?",
    "How does RAG reduce hallucinations?",
]

for query in QUERIES:
    section(f"Query: {query!r}")

    # Retrieve top-3 relevant chunks
    retrieved = vector_store.search(query, k=3)

    print(f"\n  Retrieved {len(retrieved)} chunks:\n")
    for i, chunk in enumerate(retrieved, 1):
        preview = chunk.text[:90].replace("\n", " ")
        print(f"  [{i}] id={chunk.id}  doc={chunk.doc_id}  tokens={chunk.token_count}")
        print(f"       {preview!r}\n")

    # Simulate answer citing the top chunk
    answer_chunk = retrieved[0]
    print(f"  Simulated LLM answer (citing chunk {answer_chunk.id!r}):")
    print(f"  Based on the retrieved context, {answer_chunk.text[:120].replace(chr(10), ' ')}...")

    # ── Provenance trace ──────────────────────────────────────────────────────
    print(f"\n  Provenance trace for cited chunk {answer_chunk.id!r}:")
    origin = store.trace(answer_chunk.id)
    print(f"    doc_id:      {origin.doc_id}")
    print(f"    span:        {origin.span}  (char offsets in original file)")
    print(f"    token_count: {origin.token_count}")
    print(f"    strategy:    {origin.strategy.value}")
    print(f"    merged_from: {origin.merged_from}")
    print(f"\n    Source passage:")
    print(f"    \"{origin.text}\"")


# ── Step 3: Show all chunks for a specific document ───────────────────────────

section("Step 3 — retrieve all chunks from a document")

doc_chunks = store.trace_doc("bert.txt")
print(f"\n  All {len(doc_chunks)} chunks from bert.txt:\n")
for c in doc_chunks:
    print(f"  [{c.chunk_index}] id={c.id}  span={c.span}  tokens={c.token_count}")
    print(f"       {c.text[:80].replace(chr(10), ' ')!r}\n")

print("RAG pipeline demo complete.")
