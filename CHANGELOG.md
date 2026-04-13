# Changelog

## [0.1.0] — 2026-04-13

### Added
- `Chunker` class with `fixed`, `recursive`, `structural`, `semantic`, `adaptive` strategies
- `ChunkResult` dataclass with stable ID, span, token_count, strategy, merged_from, metadata
- `ProvenanceStore` — SQLite-backed chunk registry with `register`, `trace`, `trace_doc`
- `benchmark()` — compare strategies on your corpus with coherence scoring
- `BatchChunker` — concurrent processing with checkpoint/resume
- `chunkops` CLI with `chunk` and `bench` subcommands
- Full test suite (60 tests across all modules)
- Examples: `basic_usage.py`, `rag_pipeline.py`
