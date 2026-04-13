"""Core data models for chunkops."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    ADAPTIVE = "adaptive"


@dataclass
class ChunkResult:
    """
    A single chunk with full provenance metadata.

    Attributes:
        text:        The chunk text content.
        id:          Stable SHA-256 derived ID (first 12 hex chars).
        doc_id:      The source document identifier.
        chunk_index: Zero-based position in the chunk sequence.
        span:        (char_start, char_end) offsets in the original text.
        token_count: Number of tokens (via tiktoken).
        strategy:    The ChunkStrategy that produced this chunk.
        merged_from: Sentence indices that were merged to form this chunk.
        metadata:    Any extra key/value pairs you want to carry forward.
    """

    text: str
    doc_id: str
    chunk_index: int
    span: Tuple[int, int]
    token_count: int
    strategy: ChunkStrategy
    merged_from: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    id: str = field(init=False)

    def __post_init__(self):
        # Stable ID — same text + doc + position always hashes to same ID
        raw = f"{self.doc_id}:{self.chunk_index}:{self.text}"
        self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"ChunkResult(id={self.id!r}, tokens={self.token_count}, "
            f"span={self.span}, text={preview!r}...)"
        )
