"""
Semantic chunking — splits at topic-shift boundaries using sentence embeddings.

Requires: pip install chunkops[semantic]
          (sentence-transformers, numpy, scikit-learn)
"""

from __future__ import annotations
import re
from typing import List, Tuple
from chunkops.tokenizer import count_tokens


def chunk_semantic(
    text: str,
    min_tokens: int = 100,
    max_tokens: int = 512,
    threshold: float = 0.25,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[Tuple[str, int, int, List[int]]]:
    """
    Split text at points where semantic similarity between adjacent sentences
    drops below `threshold`. Enforces min/max token bounds.

    Args:
        text:            Input text.
        min_tokens:      Minimum tokens per chunk (prevents micro-fragments).
        max_tokens:      Maximum tokens per chunk.
        threshold:       Cosine-similarity drop that triggers a new chunk.
                         Lower = more chunks. Typical range: 0.15–0.40.
        embedding_model: sentence-transformers model name.

    Returns list of (chunk_text, char_start, char_end, sentence_indices).
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError(
            "Semantic chunking requires sentence-transformers and numpy.\n"
            "Install with: pip install chunkops[semantic]"
        )

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        s, s_start, s_end = sentences[0]
        return [(s, s_start, s_end, [0])]

    # Embed all sentences at once — batch is much faster than one-by-one
    model = SentenceTransformer(embedding_model)
    texts_only = [s[0] for s in sentences]
    embeddings = model.encode(texts_only, show_progress_bar=False, normalize_embeddings=True)

    # Compute cosine similarity between adjacent sentences
    # embeddings are already L2-normalised, so dot product = cosine similarity
    sims = [float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)]

    # Find split points where similarity drops significantly
    split_points = set()
    for i, sim in enumerate(sims):
        if sim < (1.0 - threshold):
            split_points.add(i + 1)

    # Build chunks from split points
    groups: List[List[int]] = []
    current: List[int] = []
    for i in range(len(sentences)):
        if i in split_points and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)

    # Enforce min/max token bounds — merge tiny groups, split huge ones
    groups = _enforce_bounds(groups, sentences, min_tokens, max_tokens)

    results: List[Tuple[str, int, int, List[int]]] = []
    for group in groups:
        chunk_sentences = [sentences[i] for i in group]
        chunk_text = " ".join(s[0] for s in chunk_sentences)
        char_start = chunk_sentences[0][1]
        char_end = chunk_sentences[-1][2]
        results.append((chunk_text, char_start, char_end, group))

    return results


def _split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split text into (sentence, char_start, char_end) tuples."""
    pattern = re.compile(r"(?<=[.!?])\s+")
    parts = pattern.split(text.strip())
    result = []
    cursor = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        start = text.find(part, cursor)
        end = start + len(part)
        result.append((part, start, end))
        cursor = end
    return result


def _enforce_bounds(
    groups: List[List[int]],
    sentences: List[Tuple[str, int, int]],
    min_tokens: int,
    max_tokens: int,
) -> List[List[int]]:
    """Merge groups below min_tokens; split groups above max_tokens."""
    # Merge pass — forward
    merged: List[List[int]] = []
    pending: List[int] = []

    for group in groups:
        chunk_text = " ".join(sentences[i][0] for i in (pending + group))
        if pending and count_tokens(chunk_text) <= max_tokens:
            pending.extend(group)
        elif pending:
            merged.append(pending)
            pending = list(group)
        else:
            pending = list(group)

    if pending:
        merged.append(pending)

    # Check each merged group for min_tokens — merge upward with next if too small
    final: List[List[int]] = []
    for group in merged:
        text = " ".join(sentences[i][0] for i in group)
        if final and count_tokens(text) < min_tokens:
            final[-1].extend(group)
        else:
            final.append(group)

    return final if final else merged
