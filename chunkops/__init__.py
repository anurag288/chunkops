"""
chunkops — semantic-aware chunking with provenance tracking.

Quick start:
    from chunkops import Chunker
    chunks = Chunker(strategy="recursive").chunk(text, doc_id="my_doc.txt")

    from chunkops import benchmark
    result = benchmark(docs=[text], strategies=["recursive", "semantic"])
    result.report()

    from chunkops import ProvenanceStore
    store = ProvenanceStore()
    store.register(chunks)
    origin = store.trace(chunks[0].id)
"""

from chunkops.models import ChunkResult, ChunkStrategy
from chunkops.chunker import Chunker
from chunkops.provenance import ProvenanceStore
from chunkops.benchmark import benchmark, BenchmarkResult
from chunkops.batch import BatchChunker

__version__ = "0.1.0"
__all__ = [
    "Chunker",
    "ChunkResult",
    "ChunkStrategy",
    "ProvenanceStore",
    "benchmark",
    "BenchmarkResult",
    "BatchChunker",
]
