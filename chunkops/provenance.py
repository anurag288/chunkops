"""
ProvenanceStore — trace any chunk_id back to its exact source passage.

Uses SQLite so there's zero infrastructure requirement.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from chunkops.models import ChunkResult


class ProvenanceStore:
    """
    Lightweight SQLite-backed store for chunk provenance.

    Usage:
        store = ProvenanceStore()              # in-memory (default)
        store = ProvenanceStore("./prov.db")   # persistent on disk

        store.register(chunks)
        origin = store.trace("3f8a1c9b2d41")
        print(origin.span)        # (0, 712)
        print(origin.doc_id)      # "my_doc.txt"
        print(origin.merged_from) # [0, 1, 2]
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        """
        Args:
            path: Path to SQLite database file. None = in-memory.
        """
        db_path = str(path) if path else ":memory:"
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_schema()

    def _create_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                span_start  INTEGER NOT NULL,
                span_end    INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                strategy    TEXT NOT NULL,
                merged_from TEXT NOT NULL,
                metadata    TEXT NOT NULL,
                text        TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
        self._conn.commit()

    def register(self, chunks: List[ChunkResult]) -> int:
        """
        Register a list of ChunkResult objects.

        Returns the number of chunks inserted (duplicates are skipped via
        INSERT OR IGNORE).
        """
        rows = [
            (
                c.id,
                c.doc_id,
                c.chunk_index,
                c.span[0],
                c.span[1],
                c.token_count,
                c.strategy.value,
                json.dumps(c.merged_from),
                json.dumps(c.metadata),
                c.text,
            )
            for c in chunks
        ]
        cursor = self._conn.executemany(
            """INSERT OR IGNORE INTO chunks
               (id, doc_id, chunk_index, span_start, span_end,
                token_count, strategy, merged_from, metadata, text)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.commit()
        return cursor.rowcount

    def trace(self, chunk_id: str) -> ChunkResult:
        """
        Retrieve a ChunkResult by its id.

        Raises:
            KeyError: if the chunk_id is not in the store.
        """
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"chunk_id {chunk_id!r} not found in ProvenanceStore")
        return self._row_to_chunk(row)

    def trace_doc(self, doc_id: str) -> List[ChunkResult]:
        """Return all chunks for a given doc_id, ordered by chunk_index."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def count(self) -> int:
        """Total number of registered chunks."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def docs(self) -> List[str]:
        """Return list of all unique doc_ids in the store."""
        rows = self._conn.execute("SELECT DISTINCT doc_id FROM chunks").fetchall()
        return [r[0] for r in rows]

    def clear(self):
        """Remove all chunks from the store."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _row_to_chunk(self, row) -> ChunkResult:
        from chunkops.models import ChunkStrategy
        (id_, doc_id, chunk_index, span_start, span_end,
         token_count, strategy, merged_from, metadata, text) = row
        c = ChunkResult(
            text=text,
            doc_id=doc_id,
            chunk_index=chunk_index,
            span=(span_start, span_end),
            token_count=token_count,
            strategy=ChunkStrategy(strategy),
            merged_from=json.loads(merged_from),
            metadata=json.loads(metadata),
        )
        return c

    def __repr__(self) -> str:
        return f"ProvenanceStore(chunks={self.count()}, docs={len(self.docs())})"
