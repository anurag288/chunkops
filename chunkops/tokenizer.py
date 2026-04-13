"""Token counting utilities using tiktoken."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


@lru_cache(maxsize=8)
def _get_encoding(model: str):
    """Cache tiktoken encodings — loading them is expensive."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text.

    Uses tiktoken when available; falls back to word-count heuristic
    (word_count * 1.33) which is accurate to within ~5% for English text.
    """
    if not text:
        return 0
    # Treat whitespace-only as empty. tiktoken may count whitespace as tokens,
    # but for chunking we want empty/whitespace passages to cost 0 tokens.
    if not text.strip():
        return 0
    if _HAS_TIKTOKEN:
        enc = _get_encoding(model)
        return len(enc.encode(text))
    # Fallback: rough heuristic — 1 token ≈ 0.75 words
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.33))
