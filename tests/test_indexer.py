"""Tests for the indexer and embedder."""

import tempfile
from pathlib import Path

import pytest

from bookbridge.database import _connect, init_db, get_book, get_stats
from bookbridge.embedder import Embedder, get_embedder, tokenise
from bookbridge.indexer import (
    _chunk_text,
    _detect_equations,
    _detect_figures,
    _file_hash,
    index_book,
    cache_book,
    load_cached_book,
)


# ── tokeniser ─────────────────────────────────────────────────────────────────


def test_tokenise_basic():
    tokens = tokenise("The quick brown fox jumps over the lazy dog")
    assert "quick" in tokens
    assert "the" not in tokens  # stop word


def test_tokenise_empty():
    assert tokenise("") == []


# ── embedder ──────────────────────────────────────────────────────────────────


def test_embedder_shape():
    emb = Embedder(dims=128)
    vec = emb.embed("heat diffusion equation")
    assert vec.shape == (128,)


def test_embedder_normalised():
    import numpy as np
    emb = Embedder(dims=128)
    emb.fit(["heat equation", "wave mechanics"])
    vec = emb.embed("heat equation")
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embedder_similarity():
    emb = Embedder()
    emb.fit(["heat diffusion", "wave mechanics", "quantum physics"])
    v1 = emb.embed("heat equation diffusion")
    v2 = emb.embed("diffusion heat")
    v3 = emb.embed("completely unrelated medieval history")
    sim_related = emb.cosine(v1, v2)
    sim_unrelated = emb.cosine(v1, v3)
    assert sim_related > sim_unrelated


def test_embedder_roundtrip():
    emb = Embedder()
    vec = emb.embed("test phrase")
    data = emb.to_bytes(vec)
    vec2 = emb.from_bytes(data)
    import numpy as np
    assert np.allclose(vec, vec2, atol=1e-6)


# ── chunker ───────────────────────────────────────────────────────────────────


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = _chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    # All words covered
    all_words = set()
    for c in chunks:
        all_words.update(c.split())
    expected = {f"word{i}" for i in range(100)}
    assert expected == all_words


def test_chunk_text_short():
    text = "short text"
    chunks = _chunk_text(text, chunk_size=100)
    assert chunks == ["short text"]


def test_chunk_text_empty():
    assert _chunk_text("") == []


# ── equation detection ────────────────────────────────────────────────────────


def test_detect_equations():
    text = r"The energy is $E = mc^2$ from Einstein's equation."
    eqs = _detect_equations(text, page=1)
    assert len(eqs) > 0
    # At least one has latex
    latex_eqs = [e for e in eqs if e.get("latex")]
    assert len(latex_eqs) > 0


def test_detect_equations_assignment():
    text = "The velocity v = a * t where a is acceleration and t is time"
    eqs = _detect_equations(text, page=2)
    assert any("v" in e["rendered_text"] for e in eqs)


# ── figure detection ──────────────────────────────────────────────────────────


def test_detect_figures():
    text = "Figure 1: Stress-strain curve for steel samples.\nTable 2: Summary of experimental results."
    figs = _detect_figures(text, page=5)
    assert len(figs) == 2
    types = {f["result_type"] for f in figs}
    assert "figure" in types
    assert "table" in types


def test_detect_figures_numbered():
    text = "Fig. 3A: Phase diagram of the compound."
    figs = _detect_figures(text, page=7)
    assert len(figs) == 1
    assert figs[0]["figure_number"] == "3A"


# ── file hash ─────────────────────────────────────────────────────────────────


def test_file_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h = _file_hash(f)
    assert len(h) == 32  # MD5 hex


def test_file_hash_deterministic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("some content")
    assert _file_hash(f) == _file_hash(f)


# ── offline cache ─────────────────────────────────────────────────────────────


def test_cache_roundtrip(tmp_path, monkeypatch):
    import bookbridge.indexer as idx_mod
    import bookbridge.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(idx_mod, "CACHE_DIR", tmp_path)
    pages = ["Page one content.", "Page two content.", "Page three."]
    cache_book("test-book-id", pages)
    loaded = load_cached_book("test-book-id")
    assert loaded == pages


# ── index_book integration ────────────────────────────────────────────────────


def test_index_plain_text(tmp_path, monkeypatch):
    import bookbridge.config as cfg_mod
    import bookbridge.indexer as idx_mod
    monkeypatch.setattr(cfg_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(idx_mod, "CACHE_DIR", tmp_path)

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = _connect(db_path)

    # Create a plain text "book"
    book_file = tmp_path / "sample.txt"
    book_file.write_text(
        "Introduction to Physics\n\n"
        "Chapter 1: Mechanics\n"
        "Newton's second law: F = ma where F is force, m is mass, and a is acceleration.\n"
        "Figure 1: Free body diagram showing all forces acting on the object.\n\n"
        "Chapter 2: Thermodynamics\n"
        "Heat transfer equation: Q = mcΔT for temperature changes in materials.\n"
        "Table 1: Thermal conductivity values for common materials.\n"
    )

    progress_calls = []
    book_id = index_book(
        conn,
        book_file,
        {"title": "Physics 101", "authors": ["Prof. Newton"], "year": 2020},
        progress_cb=lambda done, total: progress_calls.append((done, total)),
    )

    assert book_id is not None
    book = get_book(conn, book_id)
    assert book["title"] == "Physics 101"
    assert book["pages"] == 1  # plain text = 1 page

    stats = get_stats(conn)
    assert stats["books_indexed"] == 1
    assert stats["chunks_indexed"] >= 1
    # progress callback was called
    assert len(progress_calls) > 0

    conn.close()
