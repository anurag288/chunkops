"""Recursive paragraph/sentence-aware chunking strategy."""

from __future__ import annotations
import re
from typing import List, Tuple
from chunkops.tokenizer import count_tokens


def chunk_recursive(
    text: str,
    min_tokens: int = 50,
    max_tokens: int = 400,
) -> List[Tuple[str, int, int, List[int]]]:
    """
    Split text by paragraph boundaries first, then by sentences if paragraphs
    are too large. Merges tiny paragraphs to meet min_tokens floor.

    Returns list of (chunk_text, char_start, char_end, sentence_indices).
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # Collect raw units: (text, char_start, char_end)
    units: List[Tuple[str, int, int]] = []
    for para_text, p_start, p_end in paragraphs:
        tokens = count_tokens(para_text)
        if tokens <= max_tokens:
            units.append((para_text, p_start, p_end))
        else:
            # Split large paragraph into sentences
            for sent_text, s_start, s_end in _split_sentences(para_text, p_start):
                units.append((sent_text, s_start, s_end))

    # Merge units until we hit max_tokens, respecting min_tokens floor
    results: List[Tuple[str, int, int, List[int]]] = []
    buffer_texts: List[str] = []
    buffer_start: int = 0
    buffer_end: int = 0
    sent_indices: List[int] = []
    global_sent = 0

    for unit_text, u_start, u_end in units:
        candidate = " ".join(buffer_texts + [unit_text])
        if buffer_texts and count_tokens(candidate) > max_tokens:
            # Flush buffer
            results.append((" ".join(buffer_texts), buffer_start, buffer_end, list(sent_indices)))
            buffer_texts = [unit_text]
            buffer_start = u_start
            buffer_end = u_end
            sent_indices = [global_sent]
        else:
            if not buffer_texts:
                buffer_start = u_start
            buffer_texts.append(unit_text)
            buffer_end = u_end
            sent_indices.append(global_sent)
        global_sent += 1

    if buffer_texts:
        results.append((" ".join(buffer_texts), buffer_start, buffer_end, list(sent_indices)))

    return results


def _split_paragraphs(text: str) -> List[Tuple[str, int, int]]:
    """Split on double newlines, returning (text, start, end) tuples."""
    parts = re.split(r"\n\s*\n", text)
    result = []
    cursor = 0
    for part in parts:
        part = part.strip()
        if not part:
            cursor += len(part) + 2
            continue
        start = text.find(part, cursor)
        end = start + len(part)
        result.append((part, start, end))
        cursor = end
    return result


def _split_sentences(text: str, offset: int = 0) -> List[Tuple[str, int, int]]:
    """Naïve sentence split on . ! ? boundaries."""
    pattern = re.compile(r'(?<=[.!?])\s+')
    parts = pattern.split(text)
    result = []
    cursor = offset
    for part in parts:
        part = part.strip()
        if not part:
            continue
        start = offset + text.find(part, cursor - offset)
        end = start + len(part)
        result.append((part, start, end))
        cursor = end
    return result
