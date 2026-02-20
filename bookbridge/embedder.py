"""Lightweight text embedder using hashed TF vectors stored as numpy arrays.

No heavy ML dependencies: uses numpy and Python's standard library.
Provides cosine-similarity-based semantic search as a stand-in for
a full sentence-transformer while keeping the package portable.

Embeddings are fully deterministic and reproducible across process restarts:
token → dimension mapping uses MD5 (not Python's PYTHONHASHSEED-randomised
``hash()``), and term-frequency weights require no corpus ``fit()`` step.
"""

from __future__ import annotations

import hashlib
import re
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
    """Hashed TF dense vector embedder.

    Each token is mapped to a dimension via MD5 (stable across restarts,
    unlike Python's ``hash()`` which is randomised by PYTHONHASHSEED).
    Weights are plain term-frequency — no IDF step is needed, so stored
    embeddings stay valid across process restarts without persisting any
    corpus statistics.
    """

    def __init__(self, dims: int = EMBEDDING_DIMENSIONS) -> None:
        self.dims = dims

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> None:
        """No-op kept for API compatibility. Embeddings are IDF-free."""

    def embed(self, text: str) -> np.ndarray:
        """Return a normalised dense TF embedding vector for *text*."""
        tokens = tokenise(text)
        if not tokens:
            return np.zeros(self.dims, dtype=np.float32)

        tf: Counter = Counter(tokens)
        n_tokens = len(tokens)
        vec = np.zeros(self.dims, dtype=np.float32)

        for token, count in tf.items():
            tf_score = count / n_tokens
            # Use MD5 for a stable, PYTHONHASHSEED-independent token → index mapping
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims
            vec[idx] += tf_score

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
