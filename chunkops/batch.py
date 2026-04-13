"""
BatchChunker — process large document corpora with checkpoint/resume.

Usage:
    from chunkops import BatchChunker

    bc = BatchChunker(strategy="recursive", workers=4, checkpoint="./ckpt")
    results = bc.run(docs_iterator, on_progress=lambda n, t: print(f"{n}/{t}"))
"""

from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, Union

from chunkops.chunker import Chunker
from chunkops.models import ChunkResult


class BatchChunker:
    """
    Process an iterable of (doc_id, text) tuples with concurrency and
    checkpoint/resume support.

    Args:
        strategy:    Chunking strategy name.
        workers:     Number of parallel threads.
        checkpoint:  Directory path to save/load progress. None = no checkpointing.
        min_tokens:  Passed to Chunker.
        max_tokens:  Passed to Chunker.
    """

    def __init__(
        self,
        strategy: str = "recursive",
        workers: int = 4,
        checkpoint: Optional[Union[str, Path]] = None,
        min_tokens: int = 50,
        max_tokens: int = 400,
    ):
        self.strategy = strategy
        self.workers = workers
        self.checkpoint_dir = Path(checkpoint) if checkpoint else None
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._chunker = Chunker(strategy=strategy, min_tokens=min_tokens, max_tokens=max_tokens)

    def run(
        self,
        docs: Iterable[Union[Tuple[str, str], str]],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[ChunkResult]:
        """
        Process all documents.

        Args:
            docs:        Iterable of (doc_id, text) tuples, or plain strings
                         (doc_id will be auto-assigned as "doc_N").
            on_progress: Optional callback(completed, total).

        Returns:
            Flat list of all ChunkResult objects across all documents.
        """
        doc_list = list(self._normalise(docs))
        total = len(doc_list)
        completed_ids = self._load_checkpoint()

        pending = [(doc_id, text) for doc_id, text in doc_list if doc_id not in completed_ids]
        already_done = total - len(pending)

        all_results: List[ChunkResult] = []
        done_count = already_done

        if on_progress and already_done:
            on_progress(already_done, total)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_id = {
                executor.submit(self._chunker.chunk, text, doc_id): doc_id
                for doc_id, text in pending
            }
            for future in as_completed(future_to_id):
                doc_id = future_to_id[future]
                try:
                    chunks = future.result()
                    all_results.extend(chunks)
                    completed_ids.add(doc_id)
                    done_count += 1
                    if on_progress:
                        on_progress(done_count, total)
                    if self.checkpoint_dir and done_count % 100 == 0:
                        self._save_checkpoint(completed_ids)
                except Exception as exc:
                    print(f"  [warn] doc {doc_id!r} failed: {exc}")

        if self.checkpoint_dir:
            self._save_checkpoint(completed_ids)

        return all_results

    def _normalise(self, docs: Iterable) -> Iterator[Tuple[str, str]]:
        for i, item in enumerate(docs):
            if isinstance(item, str):
                yield (f"doc_{i}", item)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                yield (str(item[0]), str(item[1]))
            else:
                raise ValueError(f"docs must be (doc_id, text) tuples or plain strings, got {type(item)}")

    def _checkpoint_path(self) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / "completed_ids.pkl"

    def _load_checkpoint(self) -> set:
        if self.checkpoint_dir is None:
            return set()
        p = self._checkpoint_path()
        if p.exists():
            with open(p, "rb") as f:
                ids = pickle.load(f)
            print(f"  [checkpoint] resuming from {len(ids)} completed docs")
            return ids
        return set()

    def _save_checkpoint(self, completed_ids: set):
        with open(self._checkpoint_path(), "wb") as f:
            pickle.dump(completed_ids, f)
