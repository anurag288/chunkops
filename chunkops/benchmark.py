"""
Benchmark multiple chunking strategies on your own corpus.

Usage:
    from chunkops import benchmark
    result = benchmark(docs=[text1, text2], strategies=["recursive", "semantic"])
    result.report()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from chunkops.chunker import Chunker
from chunkops.models import ChunkStrategy
from chunkops.tokenizer import count_tokens


@dataclass
class StrategyStats:
    strategy: str
    chunk_count: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    total_tokens: int
    elapsed_ms: float
    boundary_breaks: int       # chunks smaller than min viable size (< 20 tokens)
    coherence_score: float     # mean adjacent-sentence cosine sim (0–1), -1 if unavailable


@dataclass
class BenchmarkResult:
    stats: List[StrategyStats]
    best_strategy: str
    metric: str
    doc_count: int

    def report(self, width: int = 80):
        """Print a formatted comparison table to stdout."""
        sep = "─" * width
        print(f"\n{'chunkops benchmark':^{width}}")
        print(sep)
        print(f"  docs: {self.doc_count}   metric: {self.metric}   best: {self.best_strategy}")
        print(sep)
        header = f"{'strategy':<14}{'chunks':>8}{'avg tok':>10}{'min':>7}{'max':>7}{'coherence':>12}{'breaks':>8}{'ms':>8}"
        print(header)
        print(sep)
        for s in sorted(self.stats, key=lambda x: -x.coherence_score if self.metric == "coherence" else x.avg_tokens):
            marker = " *" if s.strategy == self.best_strategy else "  "
            coh = f"{s.coherence_score:.3f}" if s.coherence_score >= 0 else "  n/a "
            print(
                f"{marker}{s.strategy:<12}{s.chunk_count:>8}{s.avg_tokens:>10.1f}"
                f"{s.min_tokens:>7}{s.max_tokens:>7}{coh:>12}{s.boundary_breaks:>8}{s.elapsed_ms:>8.1f}"
            )
        print(sep)
        print("  * = best by selected metric\n")

    def best(self) -> StrategyStats:
        return next(s for s in self.stats if s.strategy == self.best_strategy)


def benchmark(
    docs: List[str],
    strategies: Optional[List[Union[str, ChunkStrategy]]] = None,
    metric: str = "coherence",
    min_tokens: int = 50,
    max_tokens: int = 400,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> BenchmarkResult:
    """
    Benchmark chunking strategies on a list of documents.

    Args:
        docs:            List of raw text documents to test on.
        strategies:      List of strategy names. Defaults to
                         ["fixed", "recursive", "structural"].
                         Include "semantic" if chunkops[semantic] is installed.
        metric:          "coherence" | "chunk_count" | "avg_tokens"
        min_tokens:      Passed to each Chunker.
        max_tokens:      Passed to each Chunker.
        embedding_model: Used only when "semantic" is in strategies.

    Returns:
        BenchmarkResult with .report() method and .best_strategy attribute.
    """
    if strategies is None:
        strategies = ["fixed", "recursive", "structural"]

    all_stats: List[StrategyStats] = []

    for strat in strategies:
        strat_name = strat.value if isinstance(strat, ChunkStrategy) else strat
        chunker = Chunker(
            strategy=strat_name,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            embedding_model=embedding_model,
        )

        all_chunks = []
        t0 = time.perf_counter()
        for i, doc in enumerate(docs):
            try:
                chunks = chunker.chunk(doc, doc_id=f"doc_{i}")
                all_chunks.extend(chunks)
            except ImportError:
                print(f"  [skip] {strat_name}: missing optional dependency (pip install chunkops[semantic])")
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not all_chunks:
            continue

        token_counts = [c.token_count for c in all_chunks]
        coherence = _compute_coherence(all_chunks)

        all_stats.append(StrategyStats(
            strategy=strat_name,
            chunk_count=len(all_chunks),
            avg_tokens=sum(token_counts) / len(token_counts),
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            total_tokens=sum(token_counts),
            elapsed_ms=elapsed_ms,
            boundary_breaks=sum(1 for t in token_counts if t < 20),
            coherence_score=coherence,
        ))

    if not all_stats:
        raise RuntimeError("No strategies produced results.")

    best = _pick_best(all_stats, metric)
    return BenchmarkResult(stats=all_stats, best_strategy=best, metric=metric, doc_count=len(docs))


def _compute_coherence(chunks) -> float:
    """
    Compute a coherence proxy: proportion of adjacent sentence pairs within
    chunks that share vocabulary (Jaccard similarity). No embedding required.

    Returns a float 0–1. Higher is better.
    """
    if not chunks:
        return -1.0

    scores = []
    for chunk in chunks:
        sentences = [s.strip() for s in chunk.text.split(". ") if s.strip()]
        if len(sentences) < 2:
            continue
        for i in range(len(sentences) - 1):
            a = set(sentences[i].lower().split())
            b = set(sentences[i + 1].lower().split())
            union = a | b
            if union:
                scores.append(len(a & b) / len(union))

    return round(sum(scores) / len(scores), 4) if scores else 0.5


def _pick_best(stats: List[StrategyStats], metric: str) -> str:
    if metric == "coherence":
        return max(stats, key=lambda s: s.coherence_score).strategy
    if metric == "chunk_count":
        return min(stats, key=lambda s: s.chunk_count).strategy
    if metric == "avg_tokens":
        # closest to 300 tokens is usually a sweet spot
        return min(stats, key=lambda s: abs(s.avg_tokens - 300)).strategy
    return max(stats, key=lambda s: s.coherence_score).strategy
