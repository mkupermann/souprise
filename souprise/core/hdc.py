"""Built-in Hyperdimensional Computing retriever.

A dependency-free HDC retriever implemented with NumPy only. Every text is
encoded as a 10,000-bit binary hypervector; similarity search is a vectorized
XOR + popcount over the packed index. This is the default retriever and the
reference implementation of the storage and search claims documented in the
README (10,000 bits = 1,250 bytes of index per entry).

The optional JuiceHDC engine (souprise.core.pipeline.HDCRetriever) can be
swapped in via RAGConfig(retriever="juicehdc") for larger corpora.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import hashlib
import json
import logging
import re
import sqlite3
import time
from typing import Any, Dict, List

import numpy as np

from souprise.core.pipeline import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

DIMENSIONS = 10_000  # bits per hypervector
PACKED_BYTES = DIMENSIONS // 8  # 1,250 bytes per entry
INDEX_SCHEMA_VERSION = 1
# Vectors are persisted in blocks of this many rows (~82 MB per block),
# well under SQLite's 1 GB single-blob limit even for huge corpora.
_SAVE_BLOCK_ROWS = 65_536

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# NumPy >= 2.0 exposes a hardware popcount; fall back to a uint8 lookup
# table on older versions.
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens plus adjacent-token bigrams."""
    tokens = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _token_vector(token: str) -> np.ndarray:
    """Deterministic random binary hypervector for a token.

    The token's BLAKE2b digest seeds NumPy's RNG, so the same token maps to
    the same hypervector across processes and platforms.
    """
    seed = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, DIMENSIONS).astype(np.uint8)


class SimpleHDCRetriever(BaseRetriever):
    """Pure-NumPy HDC retriever (no external dependencies).

    Encoding: a text's hypervector is the bitwise majority vote (bundling)
    of its token hypervectors. Search: Hamming distance between the query
    vector and every indexed vector, computed as XOR + popcount on packed
    uint8 arrays.

    Search is exact (exhaustive) and scales linearly: top-k selection uses
    argpartition rather than a full sort, and distances are computed in
    fixed-size chunks so peak memory stays bounded regardless of corpus
    size. Additional entries can be appended with add() without re-encoding
    the existing index.

    Args:
        chunk_rows: Number of index rows per distance-computation chunk.
            Bounds the temporary memory used during search
            (chunk_rows * 1,250 bytes); the default of 65,536 keeps the
            working set around 80 MB even for million-entry corpora.
    """

    def __init__(self, chunk_rows: int = 65_536):
        self.chunk_rows = chunk_rows
        self._packed: np.ndarray = np.empty((0, PACKED_BYTES), dtype=np.uint8)
        self._entries: List[Dict[str, Any]] = []
        self._token_cache: Dict[str, np.ndarray] = {}

    @property
    def size(self) -> int:
        """Number of indexed entries."""
        return len(self._entries)

    @property
    def index_bytes(self) -> int:
        """Total bytes used by the packed hypervector index."""
        return self._packed.nbytes

    def _encode(self, text: str) -> np.ndarray:
        """Encode text into a packed 10,000-bit hypervector (1,250 bytes)."""
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(PACKED_BYTES, dtype=np.uint8)
        acc = np.zeros(DIMENSIONS, dtype=np.int32)
        for token in tokens:
            vec = self._token_cache.get(token)
            if vec is None:
                vec = _token_vector(token)
                self._token_cache[token] = vec
            acc += vec
        majority = (2 * acc > len(tokens)).astype(np.uint8)
        return np.packbits(majority)

    def index(self, entries: List[Dict[str, Any]]) -> None:
        """Index entries for retrieval, replacing any existing index.

        Args:
            entries: List of dicts with 'id', 'text', and optionally 'metadata'.
        """
        start = time.perf_counter()
        vectors = [self._encode(entry["text"]) for entry in entries]
        if vectors:
            self._packed = np.vstack(vectors)
        else:
            self._packed = np.empty((0, PACKED_BYTES), dtype=np.uint8)
        self._entries = list(entries)
        logger.info(
            "Indexed %d entries in %.2fs (%d bytes of hypervectors)",
            len(entries), time.perf_counter() - start, self._packed.nbytes,
        )

    def _positions_by_id(self) -> Dict[str, int]:
        return {str(e.get("id", i)): i for i, e in enumerate(self._entries)}

    def add(self, entries: List[Dict[str, Any]]) -> None:
        """Upsert entries into the existing index without re-encoding it.

        Entries whose id already exists replace the stored record and its
        vector; new ids are appended. Only the given entries are encoded.

        Args:
            entries: List of dicts with 'id', 'text', and optionally 'metadata'.
        """
        if not entries:
            return
        positions = self._positions_by_id()
        new_vectors: List[np.ndarray] = []
        for entry in entries:
            vector = self._encode(entry["text"])
            entry_id = str(entry.get("id", ""))
            if entry_id and entry_id in positions:
                pos = positions[entry_id]
                self._packed[pos] = vector
                self._entries[pos] = entry
            else:
                new_vectors.append(vector)
                self._entries.append(entry)
                if entry_id:
                    positions[entry_id] = len(self._entries) - 1
        if new_vectors:
            self._packed = np.vstack([self._packed, *new_vectors])

    def delete(self, ids: List[str]) -> int:
        """Remove entries by id. Returns the number of entries removed."""
        drop = set(str(i) for i in ids)
        keep = [
            i for i, e in enumerate(self._entries)
            if str(e.get("id", i)) not in drop
        ]
        removed = len(self._entries) - len(keep)
        if removed:
            self._packed = self._packed[keep]
            self._entries = [self._entries[i] for i in keep]
        return removed

    def _hamming_distances(self, query_vec: np.ndarray) -> np.ndarray:
        """Hamming distances from the query to every indexed vector.

        Computed chunk-wise so temporary memory is bounded by
        chunk_rows * 1,250 bytes independent of corpus size.
        """
        n = self._packed.shape[0]
        distances = np.empty(n, dtype=np.int64)
        for start in range(0, n, self.chunk_rows):
            block = np.bitwise_xor(self._packed[start:start + self.chunk_rows], query_vec)
            if _HAS_BITWISE_COUNT:
                counts = np.bitwise_count(block)
            else:
                counts = _POPCOUNT8[block]
            distances[start:start + self.chunk_rows] = counts.sum(axis=1)
        return distances

    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Return the top-k entries by hypervector similarity.

        Similarity is 1 - normalized Hamming distance, in [0, 1]. Top-k
        selection uses argpartition (O(n)) instead of a full sort.
        """
        if not self._entries:
            raise RuntimeError("Retriever not initialized. Call index() first.")

        query_vec = self._encode(query)
        distances = self._hamming_distances(query_vec)
        k = min(k, len(self._entries))
        top = np.argpartition(distances, k - 1)[:k]
        top = top[np.argsort(distances[top])]

        results = []
        for idx in top:
            entry = self._entries[idx]
            results.append(RetrievalResult(
                title=str(entry.get("id", idx)),
                content=entry["text"],
                score=float(1.0 - distances[idx] / DIMENSIONS),
                metadata=entry.get("metadata", {}) or {},
            ))
        return results

    def clear(self) -> None:
        """Clear the index."""
        self._packed = np.empty((0, PACKED_BYTES), dtype=np.uint8)
        self._entries = []
        self._token_cache = {}

    def save(self, path: str) -> None:
        """Persist the index to a SQLite file.

        Stores entries (id, text, metadata) and the packed hypervector
        matrix in row blocks. The file is self-contained; load() restores
        an identical retriever without re-encoding anything.
        """
        con = sqlite3.connect(path)
        try:
            con.executescript(
                "DROP TABLE IF EXISTS meta;"
                "DROP TABLE IF EXISTS entries;"
                "DROP TABLE IF EXISTS vector_blocks;"
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
                "CREATE TABLE entries (pos INTEGER PRIMARY KEY, id TEXT,"
                " text TEXT, metadata TEXT);"
                "CREATE TABLE vector_blocks (block INTEGER PRIMARY KEY, data BLOB);"
            )
            con.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("schema_version", str(INDEX_SCHEMA_VERSION)),
                ("dimensions", str(DIMENSIONS)),
                ("entry_count", str(len(self._entries))),
            ])
            con.executemany(
                "INSERT INTO entries VALUES (?, ?, ?, ?)",
                (
                    (i, str(e.get("id", i)), e["text"],
                     json.dumps(e.get("metadata") or {}))
                    for i, e in enumerate(self._entries)
                ),
            )
            con.executemany(
                "INSERT INTO vector_blocks VALUES (?, ?)",
                (
                    (b, self._packed[start:start + _SAVE_BLOCK_ROWS].tobytes())
                    for b, start in enumerate(
                        range(0, self._packed.shape[0], _SAVE_BLOCK_ROWS))
                ),
            )
            con.commit()
        finally:
            con.close()
        logger.info("Saved %d entries to %s", len(self._entries), path)

    @classmethod
    def load(cls, path: str, chunk_rows: int = 65_536) -> "SimpleHDCRetriever":
        """Load an index previously written by save().

        Restores entries and the packed hypervector matrix without
        re-encoding, so loading a million-entry index takes seconds,
        not minutes.
        """
        con = sqlite3.connect(path)
        try:
            meta = dict(con.execute("SELECT key, value FROM meta"))
            if int(meta.get("dimensions", -1)) != DIMENSIONS:
                raise ValueError(
                    f"Index at {path} uses {meta.get('dimensions')} bit vectors; "
                    f"this build expects {DIMENSIONS}."
                )
            retriever = cls(chunk_rows=chunk_rows)
            retriever._entries = [
                {"id": entry_id, "text": text, "metadata": json.loads(metadata)}
                for entry_id, text, metadata in con.execute(
                    "SELECT id, text, metadata FROM entries ORDER BY pos")
            ]
            blocks = [
                np.frombuffer(data, dtype=np.uint8).reshape(-1, PACKED_BYTES)
                for (data,) in con.execute(
                    "SELECT data FROM vector_blocks ORDER BY block")
            ]
            if blocks:
                retriever._packed = np.vstack(blocks)
            expected = int(meta.get("entry_count", len(retriever._entries)))
            if retriever._packed.shape[0] != expected:
                raise ValueError(
                    f"Index at {path} is corrupt: {retriever._packed.shape[0]} "
                    f"vectors for {expected} entries."
                )
        finally:
            con.close()
        logger.info("Loaded %d entries from %s", len(retriever._entries), path)
        return retriever
