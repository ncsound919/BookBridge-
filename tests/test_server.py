"""Tests for HTTP server endpoints."""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import bookbridge.config as cfg
import bookbridge.server as srv_mod
import bookbridge.database as db_mod
from bookbridge.database import init_db, _connect, upsert_book, insert_chunk
from bookbridge.embedder import get_embedder


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets its own database and cache directory."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(srv_mod, "DB_PATH", db_path)

    init_db(db_path)
    conn = _connect(db_path)
    monkeypatch.setattr(srv_mod, "_db_conn", conn)

    # Reset embedder
    import bookbridge.embedder as emb_mod
    monkeypatch.setattr(emb_mod, "_embedder", None)

    yield conn
    conn.close()
    monkeypatch.setattr(srv_mod, "_db_conn", None)


@pytest.fixture
def client(isolated_db):
    return TestClient(srv_mod.app, raise_server_exceptions=True)


@pytest.fixture
def book_id(isolated_db):
    """Insert a sample book with one chunk into the test DB."""
    conn = isolated_db
    with conn:
        bid = upsert_book(
            conn,
            {
                "title": "Physics Fundamentals",
                "authors": ["Isaac Newton"],
                "year": 1687,
                "publisher": "Royal Society",
                "subject_areas": ["physics"],
                "tags": ["classic"],
            },
        )
        emb = get_embedder()
        text = "Newton's laws of motion describe the relationship between forces and motion."
        emb.fit([text])
        vec = emb.embed(text)
        insert_chunk(conn, bid, 0, text, 1, 1, "Laws of Motion", emb.to_bytes(vec))
    return bid


# ── health ────────────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "books_indexed" in data


# ── books ─────────────────────────────────────────────────────────────────────


def test_list_books_empty(client):
    r = client.get("/books")
    assert r.status_code == 200
    assert r.json() == []


def test_list_books(client, book_id):
    r = client.get("/books")
    assert r.status_code == 200
    books = r.json()
    assert len(books) == 1
    assert books[0]["book_id"] == book_id
    assert books[0]["title"] == "Physics Fundamentals"


def test_book_detail(client, book_id):
    r = client.get(f"/books/{book_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == book_id
    assert data["title"] == "Physics Fundamentals"


def test_book_detail_not_found(client):
    r = client.get("/books/nonexistent-id")
    assert r.status_code == 404


def test_list_books_filter_tag(client, book_id):
    r = client.get("/books?tag=classic")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = client.get("/books?tag=nonexistent")
    assert r.status_code == 200
    assert len(r.json()) == 0


# ── search ────────────────────────────────────────────────────────────────────


def test_search_returns_results(client, book_id):
    r = client.post("/search", json={"query": "Newton motion laws"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "query_ms" in data


def test_search_keyword_mode(client, book_id):
    r = client.post("/search", json={"query": "Newton motion laws", "search_mode": "keyword"})
    assert r.status_code == 200


def test_search_semantic_mode(client, book_id):
    r = client.post("/search", json={"query": "force and motion", "search_mode": "semantic"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_search_with_filters(client, book_id):
    r = client.post(
        "/search",
        json={
            "query": "physics",
            "filters": {"tags": ["classic"]},
        },
    )
    assert r.status_code == 200


def test_search_min_score_filter(client, book_id):
    r = client.post(
        "/search",
        json={"query": "Newton motion", "min_score": 999.0},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["results"] == []


# ── citation ──────────────────────────────────────────────────────────────────


def test_citation_apa(client, book_id):
    r = client.post("/citation", json={"book_id": book_id, "style": "APA"})
    assert r.status_code == 200
    data = r.json()
    assert "citation" in data
    assert "Newton" in data["citation"]


def test_citation_bibtex(client, book_id):
    r = client.post("/citation", json={"book_id": book_id, "style": "BibTeX"})
    assert r.status_code == 200
    data = r.json()
    assert data["bibtex_key"] is not None
    assert "@book" in data["citation"]


def test_citation_not_found(client):
    r = client.post("/citation", json={"book_id": "nonexistent", "style": "APA"})
    assert r.status_code == 404


# ── link_activity ──────────────────────────────────────────────────────────────


def test_link_activity(client, book_id):
    r = client.post(
        "/link_activity",
        json={
            "activity_type": "file",
            "activity_id": "solver.py",
            "agent_or_process_id": "agent-test",
            "references": [
                {"book_id": book_id, "page_start": 1, "page_end": 5, "reason": "Used Newton's 2nd law"}
            ],
        },
    )
    assert r.status_code == 201


def test_link_activity_bad_book(client):
    r = client.post(
        "/link_activity",
        json={
            "activity_type": "file",
            "activity_id": "test.py",
            "references": [{"book_id": "nonexistent-id"}],
        },
    )
    assert r.status_code == 400


# ── reading_plan ──────────────────────────────────────────────────────────────


def test_reading_plan(client, book_id):
    r = client.post(
        "/reading_plan",
        json={"topic": "Newton laws of motion", "goal": "implement a physics engine"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "plan" in data
    assert data["topic"] == "Newton laws of motion"


# ── add_book ──────────────────────────────────────────────────────────────────


def test_add_book(client, tmp_path, monkeypatch):
    import bookbridge.indexer as idx_mod
    monkeypatch.setattr(idx_mod, "CACHE_DIR", tmp_path)

    book_file = tmp_path / "mybook.txt"
    book_file.write_text(
        "Chapter 1\nThis is a test book about science and discovery. "
        "It contains information about important phenomena."
    )
    r = client.post(
        "/books/add",
        json={
            "local_path": str(book_file),
            "title": "Science Discovery",
            "authors": ["Dr. Scientist"],
            "year": 2024,
        },
    )
    assert r.status_code == 201
    data = r.json()
    assert "book_id" in data


def test_add_book_not_found(client):
    r = client.post(
        "/books/add",
        json={"local_path": "/nonexistent/path/book.pdf"},
    )
    assert r.status_code == 404


# ── retrieve ──────────────────────────────────────────────────────────────────


def test_retrieve_from_chunks(client, book_id):
    r = client.post(
        "/retrieve",
        json={"book_id": book_id, "page_start": 1, "page_end": 1},
    )
    assert r.status_code == 200


def test_retrieve_not_found(client):
    r = client.post(
        "/retrieve",
        json={"book_id": "nonexistent", "page_start": 1, "page_end": 1},
    )
    assert r.status_code == 404


# ── equations & figures ───────────────────────────────────────────────────────


def test_equations_empty(client, book_id):
    r = client.post("/equations", json={"query": "energy equation"})
    assert r.status_code == 200
    assert "results" in r.json()


def test_figures_empty(client, book_id):
    r = client.post("/figures", json={"query": "stress strain curve"})
    assert r.status_code == 200
    assert "results" in r.json()


# ── graph/related ──────────────────────────────────────────────────────────────


def test_graph_related(client, book_id):
    r = client.post(
        "/graph/related",
        json={"seed": {"type": "book_id", "value": book_id}, "max_hops": 1},
    )
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
