"""Lightweight text embedder using TF-IDF vectors stored as numpy arrays.

No heavy ML dependencies: uses numpy and Python's standard library.
Provides cosine-similarity-based semantic search as a stand-in for
a full sentence-transformer while keeping the package portable.
"""

from __future__ import annotations

import math
import re
import struct
from collections import Counter
from typing import Optional

import numpy as np

from .config import EMBEDDING_DIMENSIONS

# ── tokenisation ─────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that", "these",
    "those", "as", "if", "then", "than", "so", "also", "not", "no", "nor",
    "yet", "both", "either", "each", "any", "all", "other", "such",
}


def tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ── vocabulary / IDF table ────────────────────────────────────────────────────

class Embedder:
    """TF-IDF-inspired dense vector embedder.

    Vocabulary is built lazily from ingested documents and hashed into a
    fixed-size ``EMBEDDING_DIMENSIONS``-dimensional space so that the
    model never needs re-training when new books are added.
    """

    def __init__(self, dims: int = EMBEDDING_DIMENSIONS) -> None:
        self.dims = dims
        # IDF weights: token -> log(1 + N / (1 + df))
        self._doc_freq: Counter = Counter()
        self._total_docs: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> None:
        """Update IDF table from a batch of documents."""
        self._total_docs += len(texts)
        for text in texts:
            unique_tokens = set(tokenise(text))
            self._doc_freq.update(unique_tokens)

    def embed(self, text: str) -> np.ndarray:
        """Return a normalised dense embedding vector for *text*."""
        tokens = tokenise(text)
        if not tokens:
            return np.zeros(self.dims, dtype=np.float32)

        tf: Counter = Counter(tokens)
        n_tokens = len(tokens)
        vec = np.zeros(self.dims, dtype=np.float32)

        for token, count in tf.items():
            tf_score = count / n_tokens
            df = self._doc_freq.get(token, 0)
            idf = math.log(1.0 + (self._total_docs + 1) / (df + 1))
            weight = tf_score * idf
            # Hash token into dims-dimensional space
            idx = hash(token) % self.dims
            vec[idx] += weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def to_bytes(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32).copy()

    # ── similarity ────────────────────────────────────────────────────────────

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# Module-level singleton so the IDF table is shared across the process
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
