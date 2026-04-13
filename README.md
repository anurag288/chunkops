# chunkops

**Semantic-aware chunking with provenance tracking for production RAG and LLM data pipelines.**

```
pip install chunkops
```

---

## The problem

Every RAG system reinvents chunking from scratch. Existing tools (LangChain splitters, LlamaIndex nodes) are buried inside their frameworks, don't interop, and return plain strings with no provenance. When your pipeline gives a wrong answer three weeks after deploy, you have no way to trace which chunk caused it.

`chunkops` solves three things:

1. **Standalone chunking** — works with any vector DB, any LLM, any framework
2. **Provenance** — every `ChunkResult` carries a stable ID, character span, and lineage so you can trace any LLM output back to its exact source passage
3. **Benchmarking** — run 3–4 strategies on your actual corpus with one function call before committing

---

## Installation

```bash
# Core — fixed, recursive, structural strategies + provenance + benchmark
pip install llm-chunk-optimizer

# Add semantic chunking (embedding-based topic boundaries)
pip install "llm-chunk-optimizer[semantic]"

# Everything + dev/test tools
pip install "llm-chunk-optimizer[dev]"
```

### Import name vs install name

- Install name: `llm-chunk-optimizer`
- Import name: `chunkops`

### Dependencies


| Package | When required | Why |
|---------|---------------|-----|
| `tiktoken` | always | accurate token counting |
| `sentence-transformers` | `chunkops[semantic]` only | semantic boundary detection |
| `numpy` | `chunkops[semantic]` only | vector operations |
| `scikit-learn` | `chunkops[semantic]` only | cosine similarity |
| `pytest` | `chunkops[dev]` only | running tests |

---

## Quick start

```python
from chunkops import Chunker

chunker = Chunker(strategy="recursive", min_tokens=100, max_tokens=400)
chunks = chunker.chunk(text, doc_id="my_doc.txt")

for c in chunks:
    print(c.id, c.token_count, c.span, c.text[:60])
```

---

## Strategies

| Strategy | Best for | Speed | Quality |
|----------|----------|-------|---------|
| `fixed` | quick prototyping | fastest | poor — breaks sentences |
| `recursive` | most documents | fast | good — respects paragraphs |
| `structural` | markdown, wikis, docs | fast | great — uses headings |
| `semantic` | dense unstructured prose | slow | best — topic boundaries |
| `adaptive` | mixed corpora | fast | good — auto-selects |

```python
# Semantic chunking (requires pip install chunkops[semantic])
chunker = Chunker(
    strategy="semantic",
    min_tokens=150,        # prevents micro-fragments
    max_tokens=512,
    semantic_threshold=0.25,
    embedding_model="all-MiniLM-L6-v2",
)
chunks = chunker.chunk(text, doc_id="paper.txt")
```

---

## ChunkResult — provenance fields

```python
chunk = chunks[0]

chunk.id            # "3f8a1c9b2d41"  — stable 12-char hash
chunk.doc_id        # "my_doc.txt"
chunk.chunk_index   # 0
chunk.span          # (0, 712)        — char offsets in original text
chunk.token_count   # 142
chunk.strategy      # ChunkStrategy.RECURSIVE
chunk.merged_from   # [0, 1, 2]       — sentence indices merged here
chunk.metadata      # {"source": "arxiv", "year": 2017}
chunk.text          # "The transformer architecture..."
```

---

## ProvenanceStore

Trace any LLM output back to its exact source passage. Uses SQLite — zero infrastructure required.

```python
from chunkops import ProvenanceStore

store = ProvenanceStore()              # in-memory
store = ProvenanceStore("./prov.db")   # persistent on disk

store.register(chunks)

# Later — your RAG pipeline cites a chunk_id
origin = store.trace("3f8a1c9b2d41")
print(origin.doc_id)        # "my_doc.txt"
print(origin.span)          # (0, 712)
print(origin.merged_from)   # [0, 1, 2]
print(origin.text)          # exact source passage

# All chunks from a document
doc_chunks = store.trace_doc("my_doc.txt")

# Stats
print(store.count())        # 487
print(store.docs())         # ["my_doc.txt", "paper2.txt", ...]
```

---

## Benchmark

Find the best strategy for your corpus before committing.

```python
from chunkops import benchmark

result = benchmark(
    docs=[doc1, doc2, doc3],
    strategies=["fixed", "recursive", "structural"],
    metric="coherence",     # or "chunk_count", "avg_tokens"
)
result.report()
# ── chunkops benchmark ──────────────────────────────────────────────────────
#   docs: 3   metric: coherence   best: recursive
# ────────────────────────────────────────────────────────────────────────────
#   strategy       chunks   avg tok    min    max   coherence  breaks      ms
# ────────────────────────────────────────────────────────────────────────────
#   fixed              18      47.2     12    112       0.142       4     2.1
# * recursive           9      91.4     63    147       0.381       0     1.3
#   structural          9      91.4     63    147       0.374       0     1.2
# ────────────────────────────────────────────────────────────────────────────

print(result.best_strategy)   # "recursive"
print(result.best().avg_tokens)
```

---

## BatchChunker

Process large corpora with concurrency and checkpoint/resume.

```python
from chunkops import BatchChunker

bc = BatchChunker(
    strategy="adaptive",
    workers=8,
    checkpoint="./ckpt/run_001",   # crash at doc 80k → resume from 80k
    min_tokens=150,
    max_tokens=512,
)

results = bc.run(
    docs_iterator,   # iterable of (doc_id, text) tuples, or plain strings
    on_progress=lambda n, t: print(f"{n}/{t} docs"),
)
# 10000/100000 docs
# 50000/100000 docs  [checkpoint saved]
# KeyboardInterrupt → resume → 100000/100000 docs
# Total: 487,302 chunks
```

---

## CLI

```bash
# Chunk a file
chunkops chunk my_doc.txt --strategy recursive --max-tokens 400

# Benchmark strategies on a file
chunkops bench my_doc.txt --strategies fixed,recursive,structural --metric coherence
```

---

## RAG pipeline integration

```python
from chunkops import Chunker, ProvenanceStore

# 1. Ingest
chunker = Chunker(strategy="recursive")
store = ProvenanceStore("./prov.db")

for doc_id, text in your_documents:
    chunks = chunker.chunk(text, doc_id=doc_id)
    store.register(chunks)
    your_vector_db.upsert([
        {"id": c.id, "vector": embed(c.text), "text": c.text}
        for c in chunks
    ])

# 2. Query
results = your_vector_db.search(query, k=5)

# 3. Trace
for r in results:
    origin = store.trace(r["id"])
    print(f"Source: {origin.doc_id}  span: {origin.span}")
```

---

## Development

```bash
git clone https://github.com/yourusername/chunkops
cd chunkops
pip install -e ".[dev]"
pytest
```

---

## Roadmap

- [ ] `LATE` chunking strategy (late chunking / contextual retrieval)
- [ ] LangChain `TextSplitter` adapter
- [ ] LlamaIndex `NodeParser` adapter
- [ ] Async `BatchChunker`
- [ ] Export provenance to Parquet / Arrow
- [ ] OpenTelemetry tracing integration

---

## License

MIT
