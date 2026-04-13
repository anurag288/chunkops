"""Structural chunking — splits on markdown/HTML headings and paragraph breaks."""

from __future__ import annotations
import re
from typing import List, Tuple
from chunkops.tokenizer import count_tokens


# Matches markdown headings (# Title) and common structural markers
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|={3,}|-{3,})$", re.MULTILINE)


def chunk_structural(
    text: str,
    min_tokens: int = 30,
    max_tokens: int = 600,
) -> List[Tuple[str, int, int, List[int]]]:
    """
    Split on structural boundaries: markdown headings, paragraph breaks.
    Falls back to recursive splitting when a section exceeds max_tokens.

    Returns list of (chunk_text, char_start, char_end, sentence_indices).
    """
    sections = _split_on_structure(text)
    if not sections:
        return []

    results: List[Tuple[str, int, int, List[int]]] = []
    buffer: List[str] = []
    buf_start = 0
    buf_end = 0
    sent_counter = 0

    for sec_text, s_start, s_end in sections:
        tokens = count_tokens(sec_text)

        # Section too large — split it further by paragraphs
        if tokens > max_tokens:
            if buffer:
                results.append((" ".join(buffer), buf_start, buf_end, [sent_counter]))
                buffer = []
                sent_counter += 1
            for sub in _split_paragraphs(sec_text, s_start):
                results.append((sub[0], sub[1], sub[2], [sent_counter]))
                sent_counter += 1
            continue

        candidate = " ".join(buffer + [sec_text])
        if buffer and count_tokens(candidate) > max_tokens:
            results.append((" ".join(buffer), buf_start, buf_end, [sent_counter]))
            buffer = [sec_text]
            buf_start = s_start
            buf_end = s_end
            sent_counter += 1
        else:
            if not buffer:
                buf_start = s_start
            buffer.append(sec_text)
            buf_end = s_end

    if buffer:
        results.append((" ".join(buffer), buf_start, buf_end, [sent_counter]))

    return results


def _split_on_structure(text: str) -> List[Tuple[str, int, int]]:
    """Split text on double newlines and markdown headings."""
    raw_parts = re.split(r"(\n\s*\n|(?=^#{1,6}\s))", text, flags=re.MULTILINE)
    result = []
    cursor = 0
    for part in raw_parts:
        clean = part.strip()
        if not clean:
            cursor += len(part)
            continue
        start = text.find(clean, cursor)
        end = start + len(clean)
        result.append((clean, start, end))
        cursor = end
    return result


def _split_paragraphs(text: str, offset: int) -> List[Tuple[str, int, int]]:
    """Split a single large section into paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    result = []
    cursor = offset
    for part in parts:
        clean = part.strip()
        if not clean:
            continue
        start = offset + text.find(clean, cursor - offset)
        end = start + len(clean)
        result.append((clean, start, end))
        cursor = end
    return result
