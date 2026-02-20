"""Tests for database layer."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from bookbridge.database import (
    init_db,
    upsert_book,
    get_book,
    list_books,
    insert_chunk,
    insert_equation,
    insert_figure,
    insert_activity_reference,
    get_stats,
    create_index_job,
    get_index_job,
    update_index_job,
    get_related_nodes,
    insert_graph_edge,
    _connect,
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = _connect(db_path)
    yield conn
    conn.close()


def _sample_book(overrides=None):
    b = {
        "title": "Test Book",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2023,
        "publisher": "Test Press",
        "isbn": "978-0-000-00000-0",
        "doi": "10.0000/test",
        "subject_areas": ["physics"],
        "tags": ["science", "intro"],
        "allowed_agents": [],
    }
    if overrides:
        b.update(overrides)
    return b


# ── books ─────────────────────────────────────────────────────────────────────


def test_upsert_and_get_book(db):
    with db:
        book_id = upsert_book(db, _sample_book())
    book = get_book(db, book_id)
    assert book is not None
    assert book["title"] == "Test Book"
    assert "Alice Smith" in book["authors"]
    assert book["year"] == 2023


def test_upsert_updates_existing(db):
    meta = _sample_book()
    with db:
        book_id = upsert_book(db, meta)
    meta["id"] = book_id
    meta["title"] = "Updated Title"
    with db:
        upsert_book(db, meta)
    book = get_book(db, book_id)
    assert book["title"] == "Updated Title"


def test_get_book_not_found(db):
    assert get_book(db, "nonexistent-id") is None


def test_list_books_filter_tag(db):
    with db:
        upsert_book(db, _sample_book({"tags": ["science"]}))
        upsert_book(db, _sample_book({"title": "Other Book", "tags": ["history"]}))
    science_books = list_books(db, filter_tag="science")
    assert len(science_books) == 1
    assert science_books[0]["title"] == "Test Book"


def test_list_books_filter_author(db):
    with db:
        upsert_book(db, _sample_book())
        upsert_book(db, _sample_book({"title": "Other Book", "authors": ["Carol Davis"]}))
    alice_books = list_books(db, filter_author="Alice")
    assert len(alice_books) == 1


def test_list_books_filter_year(db):
    with db:
        upsert_book(db, _sample_book({"year": 2020}))
        upsert_book(db, _sample_book({"title": "New Book", "year": 2023}))
    recent = list_books(db, filter_year_from=2022)
    assert len(recent) == 1
    assert recent[0]["title"] == "New Book"


# ── chunks ────────────────────────────────────────────────────────────────────


def test_insert_chunk(db):
    with db:
        book_id = upsert_book(db, _sample_book())
        chunk_id = insert_chunk(db, book_id, 0, "Hello world of science.", 1, 1, "Introduction")
    row = db.execute("SELECT * FROM book_chunks WHERE id=?", (chunk_id,)).fetchone()
    assert row is not None
    assert row["text"] == "Hello world of science."
    assert row["page_start"] == 1


def test_insert_chunk_fts(db):
    with db:
        book_id = upsert_book(db, _sample_book())
        insert_chunk(db, book_id, 0, "quantum mechanics and wave functions", 5, 5)
    rows = db.execute(
        "SELECT rowid FROM book_chunks_fts WHERE book_chunks_fts MATCH '\"quantum\"'"
    ).fetchall()
    assert len(rows) > 0


# ── equations ─────────────────────────────────────────────────────────────────


def test_insert_and_search_equation(db):
    with db:
        book_id = upsert_book(db, _sample_book())
        eq_id = insert_equation(db, book_id, 42, "E = mc^2", r"$E = mc^2$", "Energy")
    rows = db.execute(
        "SELECT rowid FROM equations_fts WHERE equations_fts MATCH '\"energy\"'"
    ).fetchall()
    assert len(rows) > 0


# ── figures ───────────────────────────────────────────────────────────────────


def test_insert_and_search_figure(db):
    with db:
        book_id = upsert_book(db, _sample_book())
        fig_id = insert_figure(db, book_id, 10, "1", "Stress-strain curve", "figure")
    rows = db.execute(
        "SELECT rowid FROM figures_fts WHERE figures_fts MATCH '\"stress\"'"
    ).fetchall()
    assert len(rows) > 0


# ── activity references ───────────────────────────────────────────────────────


def test_insert_activity_reference(db):
    with db:
        book_id = upsert_book(db, _sample_book())
        ref_id = insert_activity_reference(
            db,
            activity_type="file",
            activity_id="myfile.py",
            book_id=book_id,
            page_start=1,
            page_end=5,
            reason="Used formula",
            agent_or_process_id="agent-001",
        )
    row = db.execute("SELECT * FROM activity_references WHERE id=?", (ref_id,)).fetchone()
    assert row is not None
    assert row["activity_id"] == "myfile.py"
    assert row["reason"] == "Used formula"


# ── stats ─────────────────────────────────────────────────────────────────────


def test_get_stats(db):
    with db:
        book_id = upsert_book(db, _sample_book())
        insert_chunk(db, book_id, 0, "text", 1, 1)
        insert_equation(db, book_id, 1, "x = 1")
        insert_figure(db, book_id, 2, "1", "A figure caption")
    stats = get_stats(db)
    assert stats["books_indexed"] == 1
    assert stats["chunks_indexed"] == 1
    assert stats["equations_indexed"] == 1
    assert stats["figures_indexed"] == 1


# ── index jobs ────────────────────────────────────────────────────────────────


def test_index_job_lifecycle(db):
    with db:
        job_id = create_index_job(db, ["book-1", "book-2"])
    job = get_index_job(db, job_id)
    assert job["status"] == "queued"
    assert "book-1" in job["book_ids"]
    with db:
        update_index_job(db, job_id, "done", {"percent": 100})
    job = get_index_job(db, job_id)
    assert job["status"] == "done"


# ── graph ─────────────────────────────────────────────────────────────────────


def test_graph_edges(db):
    with db:
        book_id1 = upsert_book(db, _sample_book({"title": "Book A"}))
        book_id2 = upsert_book(db, _sample_book({"title": "Book B"}))
        insert_graph_edge(db, book_id1, "book", book_id2, "book", "cites")
    nodes, edges = get_related_nodes(db, book_id1)
    edge_pairs = {(e["from_id"], e["to_id"]) for e in edges}
    assert (book_id1, book_id2) in edge_pairs
