"""Fixed-size chunking strategy."""

from __future__ import annotations
from typing import List, Tuple
from chunkops.tokenizer import count_tokens


def chunk_fixed(
    text: str,
    chunk_size: int = 200,
    overlap: int = 20,
) -> List[Tuple[str, int, int, List[int]]]:
    """
    Split text into fixed token-size windows with optional overlap.

    Returns list of (chunk_text, char_start, char_end, sentence_indices).
    """
    words = text.split()
    if not words:
        return []

    results = []
    word_positions = _word_positions(text)
    i = 0
    sent_idx = 0

    while i < len(words):
        window = []
        token_count = 0
        j = i
        while j < len(words) and token_count < chunk_size:
            window.append(words[j])
            token_count = count_tokens(" ".join(window))
            j += 1

        chunk_text = " ".join(window)
        char_start = word_positions[i][0]
        char_end = word_positions[min(j - 1, len(words) - 1)][1]
        sentence_indices = list(range(sent_idx, sent_idx + len(window)))

        results.append((chunk_text, char_start, char_end, sentence_indices))

        # Advance by (chunk_size - overlap) tokens worth of words
        overlap_words = max(1, overlap)
        i = max(i + 1, j - overlap_words)
        sent_idx += len(window) - overlap_words

    return results


def _word_positions(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) char positions for each word in text."""
    positions = []
    idx = 0
    for word in text.split():
        start = text.index(word, idx)
        end = start + len(word)
        positions.append((start, end))
        idx = end
    return positions
