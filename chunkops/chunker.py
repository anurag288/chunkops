"""Main Chunker class — the primary interface for chunkops."""

from __future__ import annotations

from typing import List, Optional, Union

from chunkops.models import ChunkResult, ChunkStrategy
from chunkops.tokenizer import count_tokens


class Chunker:
    """
    Framework-agnostic document chunker with provenance metadata.

    Strategies:
        "fixed"      — fixed token-window with overlap (fast, poor quality)
        "recursive"  — paragraph → sentence hierarchy (good default)
        "structural" — markdown/heading-aware (best for docs/wikis)
        "semantic"   — embedding-based topic boundaries (best quality,
                        requires pip install chunkops[semantic])
        "adaptive"   — heuristically picks the best strategy per document

    Examples:
        >>> chunker = Chunker(strategy="recursive", min_tokens=100, max_tokens=400)
        >>> chunks = chunker.chunk(text, doc_id="my_doc.txt")
        >>> print(chunks[0])
        ChunkResult(id='3f8a1c...', tokens=142, span=(0, 712), text='The transformer...')
    """

    def __init__(
        self,
        strategy: Union[str, ChunkStrategy] = "recursive",
        min_tokens: int = 50,
        max_tokens: int = 400,
        overlap: int = 20,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.25,
    ):
        self.strategy = ChunkStrategy(strategy)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold

    def chunk(self, text: str, doc_id: str = "document", metadata: Optional[dict] = None) -> List[ChunkResult]:
        """
        Chunk a single document.

        Args:
            text:     Raw text to chunk.
            doc_id:   Identifier for this document (used in provenance).
            metadata: Optional dict attached to every ChunkResult.

        Returns:
            List of ChunkResult objects, each with id, span, token_count,
            strategy, merged_from, and metadata fields.
        """
        if not text or not text.strip():
            return []

        strategy = self._resolve_adaptive(text) if self.strategy == ChunkStrategy.ADAPTIVE else self.strategy
        raw_chunks = self._run_strategy(strategy, text)

        results: List[ChunkResult] = []
        for idx, (chunk_text, char_start, char_end, sent_indices) in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            tokens = count_tokens(chunk_text)
            results.append(
                ChunkResult(
                    text=chunk_text.strip(),
                    doc_id=doc_id,
                    chunk_index=idx,
                    span=(char_start, char_end),
                    token_count=tokens,
                    strategy=strategy,
                    merged_from=sent_indices,
                    metadata=metadata or {},
                )
            )

        return results

    def _run_strategy(self, strategy: ChunkStrategy, text: str):
        if strategy == ChunkStrategy.FIXED:
            from chunkops.strategies.fixed import chunk_fixed
            return chunk_fixed(text, chunk_size=self.max_tokens, overlap=self.overlap)

        if strategy == ChunkStrategy.RECURSIVE:
            from chunkops.strategies.recursive import chunk_recursive
            return chunk_recursive(text, min_tokens=self.min_tokens, max_tokens=self.max_tokens)

        if strategy == ChunkStrategy.STRUCTURAL:
            from chunkops.strategies.structural import chunk_structural
            return chunk_structural(text, min_tokens=self.min_tokens, max_tokens=self.max_tokens)

        if strategy == ChunkStrategy.SEMANTIC:
            from chunkops.strategies.semantic import chunk_semantic
            return chunk_semantic(
                text,
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                threshold=self.semantic_threshold,
                embedding_model=self.embedding_model,
            )

        raise ValueError(f"Unknown strategy: {strategy}")

    def _resolve_adaptive(self, text: str) -> ChunkStrategy:
        """
        Heuristically pick the best strategy for this document.

        Rules:
        - Has markdown headings → STRUCTURAL
        - Short doc (< 300 tokens) → RECURSIVE
        - Long dense prose → prefer RECURSIVE (semantic is expensive without
          consistent gains on short corpora per FloTorch 2026 benchmarks)
        - Default → RECURSIVE
        """
        import re
        has_headings = bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))
        if has_headings:
            return ChunkStrategy.STRUCTURAL

        token_count = count_tokens(text)
        if token_count < 300:
            return ChunkStrategy.RECURSIVE

        # Check paragraph regularity — if text has clean double-newline structure,
        # recursive will be fast and effective
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) >= 3:
            return ChunkStrategy.RECURSIVE

        return ChunkStrategy.RECURSIVE
