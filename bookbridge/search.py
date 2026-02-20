"""Hybrid search (keyword FTS5 + semantic cosine similarity) over the book index."""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

from .database import get_all_chunks_with_embeddings, get_book
from .embedder import get_embedder

# ── internal helpers ──────────────────────────────────────────────────────────


def _fts_escape(q: str) -> str:
    """Wrap each token in double quotes for FTS5 safety."""
    tokens = q.strip().split()
    return " ".join(f'"{t}"' for t in tokens if t) if tokens else '""'


def _keyword_search(
    conn: sqlite3.Connection,
    query: str,
    max_results: int,
    filters: Optional[dict] = None,
) -> list[dict]:
    """Full-text search using SQLite FTS5."""
    fts_query = _fts_escape(query)
    sql = """
        SELECT c.id as chunk_id, c.book_id, c.text, c.page_start, c.page_end,
               c.section_heading, b.title as book_title, b.authors, b.year,
               b.drive_web_view_link,
               bm25(book_chunks_fts) as bm25_score
        FROM book_chunks c
        JOIN book_chunks_fts fts ON fts.rowid = c.rowid
        JOIN books b ON c.book_id = b.id
        WHERE book_chunks_fts MATCH ?
    """
    params: list = [fts_query]
    sql, params = _apply_filters(sql, params, filters)
    sql += " ORDER BY bm25_score LIMIT ?"
    params.append(max_results)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        rows = []
    return [dict(r) for r in rows]


def _semantic_search(
    conn: sqlite3.Connection,
    query: str,
    max_results: int,
    filters: Optional[dict] = None,
) -> list[dict]:
    """Cosine-similarity search against stored embeddings."""
    embedder = get_embedder()
    q_vec = embedder.embed(query)

    chunks = get_all_chunks_with_embeddings(conn)
    if not chunks:
        return []

    scored = []
    for c in chunks:
        if c.get("embedding") is None:
            continue
        c_vec = embedder.from_bytes(c["embedding"])
        score = embedder.cosine(q_vec, c_vec)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, c in scored[:max_results]:
        book = get_book(conn, c["book_id"])
        if book is None:
            continue
        # Apply filters
        if not _passes_filters(book, filters):
            continue
        results.append(
            {
                "chunk_id": c["id"],
                "book_id": c["book_id"],
                "text": c["text"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "section_heading": c["section_heading"],
                "book_title": book["title"],
                "authors": book["authors"],
                "year": book.get("year"),
                "drive_web_view_link": book.get("drive_web_view_link", ""),
                "score": score,
            }
        )
    return results[:max_results]


def _apply_filters(sql: str, params: list, filters: Optional[dict]) -> tuple[str, list]:
    if not filters:
        return sql, params
    if filters.get("book_ids"):
        placeholders = ",".join("?" * len(filters["book_ids"]))
        sql += f" AND c.book_id IN ({placeholders})"
        params.extend(filters["book_ids"])
    return sql, params


def _passes_filters(book: dict, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    if filters.get("tags"):
        if not any(t in book.get("tags", []) for t in filters["tags"]):
            return False
    if filters.get("authors"):
        book_authors = [a.lower() for a in book.get("authors", [])]
        if not any(
            any(fa.lower() in ba for ba in book_authors)
            for fa in filters["authors"]
        ):
            return False
    year_range = filters.get("year_range")
    if year_range and len(year_range) == 2:
        year = book.get("year") or 0
        if year < year_range[0] or year > year_range[1]:
            return False
    if filters.get("subject_areas"):
        book_subjects = [s.lower() for s in book.get("subject_areas", [])]
        filter_terms = [s.lower() for s in filters["subject_areas"]]
        if not any(f in bs for f in filter_terms for bs in book_subjects):
            return False
    return True


def _make_citation(book: dict, page_start: Optional[int] = None, page_end: Optional[int] = None) -> str:
    authors = book.get("authors", [])
    author_str = authors[0] if len(authors) == 1 else (", ".join(authors[:2]) + " et al." if len(authors) > 2 else " & ".join(authors))
    year = book.get("year", "n.d.")
    title = book.get("title", "Unknown")
    page_part = ""
    if page_start:
        page_part = f", p. {page_start}" if page_start == page_end else f", pp. {page_start}–{page_end}"
    return f"{author_str} ({year}). {title}{page_part}."


# ── public search function ────────────────────────────────────────────────────


def search(
    conn: sqlite3.Connection,
    query: str,
    max_results: int = 10,
    search_mode: str = "hybrid",
    include_equations: bool = False,
    include_figures: bool = False,
    filters: Optional[dict] = None,
    agent_id: Optional[str] = None,
) -> dict:
    """Run a search and return a structured response dict."""
    import time

    start = time.time()

    # Check agent access
    allowed_books = _get_allowed_books(conn, agent_id)

    if search_mode == "keyword":
        raw = _keyword_search(conn, query, max_results * 2, filters)
        results = _format_keyword_results(conn, raw, max_results, allowed_books)
    elif search_mode == "semantic":
        raw = _semantic_search(conn, query, max_results * 2, filters)
        results = _format_semantic_results(conn, raw, max_results, allowed_books)
    else:  # hybrid
        kw = _keyword_search(conn, query, max_results, filters)
        sem = _semantic_search(conn, query, max_results, filters)
        results = _merge_hybrid(conn, kw, sem, max_results, allowed_books)

    # Optionally append equations and figures
    if include_equations:
        eq_results = _search_equations_for_agent(conn, query, max_results // 2, allowed_books)
        results.extend(eq_results)

    if include_figures:
        fig_results = _search_figures_for_agent(conn, query, max_results // 2, allowed_books)
        results.extend(fig_results)

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "results": results[:max_results],
        "total_matches": len(results),
        "query_ms": elapsed_ms,
    }


def _get_allowed_books(conn: sqlite3.Connection, agent_id: Optional[str]) -> Optional[set]:
    """Return set of book IDs that *agent_id* may access, or None (= all books)."""
    if agent_id is None:
        return None
    rows = conn.execute("SELECT id, allowed_agents FROM books").fetchall()
    allowed: set = set()
    for row in rows:
        agents = json.loads(row["allowed_agents"] or "[]")
        if not agents or agent_id in agents:
            allowed.add(row["id"])
    return allowed


def _format_keyword_results(conn, raw, max_results, allowed_books):
    results = []
    for r in raw[:max_results]:
        if allowed_books is not None and r["book_id"] not in allowed_books:
            continue
        authors = json.loads(r["authors"]) if isinstance(r["authors"], str) else r["authors"]
        book = {"title": r["book_title"], "authors": authors, "year": r.get("year")}
        results.append(
            {
                "chunk_id": r["chunk_id"],
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "authors": authors,
                "year": r.get("year"),
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "text": r["text"],
                "score": -float(r.get("bm25_score") or 0),
                "citation_ready": _make_citation(book, r["page_start"], r["page_end"]),
                "drive_web_view_link": r.get("drive_web_view_link", ""),
                "result_type": "text",
            }
        )
    return results


def _format_semantic_results(conn, raw, max_results, allowed_books):
    results = []
    for r in raw[:max_results]:
        if allowed_books is not None and r["book_id"] not in allowed_books:
            continue
        book = {"title": r["book_title"], "authors": r["authors"], "year": r.get("year")}
        results.append(
            {
                "chunk_id": r.get("chunk_id", ""),
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "authors": r["authors"],
                "year": r.get("year"),
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "text": r["text"],
                "score": r.get("score", 0.0),
                "citation_ready": _make_citation(book, r["page_start"], r["page_end"]),
                "drive_web_view_link": r.get("drive_web_view_link", ""),
                "result_type": "text",
            }
        )
    return results


def _merge_hybrid(conn, kw_raw, sem_raw, max_results, allowed_books):
    """Merge keyword and semantic results, deduplicating by chunk_id."""

    seen: set = set()
    merged = []

    # Normalise keyword scores: SQLite bm25() returns negative values where
    # more-negative = better match. Negate so higher = better, matching the
    # direction of cosine similarity scores used for semantic results.
    for r in kw_raw:
        cid = r.get("chunk_id") or r.get("id", "")
        if cid in seen:
            continue
        if allowed_books is not None and r["book_id"] not in allowed_books:
            continue
        seen.add(cid)
        authors = json.loads(r["authors"]) if isinstance(r["authors"], str) else r["authors"]
        book = {"title": r["book_title"], "authors": authors, "year": r.get("year")}
        merged.append(
            {
                "chunk_id": cid,
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "authors": authors,
                "year": r.get("year"),
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "text": r["text"],
                "score": -float(r.get("bm25_score") or 0),
                "citation_ready": _make_citation(book, r["page_start"], r["page_end"]),
                "drive_web_view_link": r.get("drive_web_view_link", ""),
                "result_type": "text",
            }
        )

    for r in sem_raw:
        cid = r.get("chunk_id", "")
        if cid in seen:
            continue
        if allowed_books is not None and r["book_id"] not in allowed_books:
            continue
        seen.add(cid)
        book = {"title": r["book_title"], "authors": r["authors"], "year": r.get("year")}
        merged.append(
            {
                "chunk_id": cid,
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "authors": r["authors"],
                "year": r.get("year"),
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "text": r["text"],
                "score": r.get("score", 0.0),
                "citation_ready": _make_citation(book, r["page_start"], r["page_end"]),
                "drive_web_view_link": r.get("drive_web_view_link", ""),
                "result_type": "text",
            }
        )

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:max_results]


def _search_equations_for_agent(conn, query, max_results, allowed_books):
    from .database import search_equations_fts
    rows = search_equations_fts(conn, _fts_escape(query), max_results=max_results)
    results = []
    for r in rows:
        if allowed_books is not None and r["book_id"] not in allowed_books:
            continue
        authors = json.loads(r["authors"]) if isinstance(r["authors"], str) else r["authors"]
        book = {"title": r["book_title"], "authors": authors, "year": r.get("year")}
        results.append(
            {
                "chunk_id": r["id"],
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "authors": authors,
                "year": r.get("year"),
                "page_start": r["page"],
                "page_end": r["page"],
                "text": r.get("latex") or r["rendered_text"],
                "score": 0.5,
                "citation_ready": _make_citation(book, r["page"], r["page"]),
                "drive_web_view_link": r.get("drive_web_view_link", ""),
                "result_type": "equation",
            }
        )
    return results


def _search_figures_for_agent(conn, query, max_results, allowed_books):
    from .database import search_figures_fts
    rows = search_figures_fts(conn, _fts_escape(query), max_results=max_results)
    results = []
    for r in rows:
        if allowed_books is not None and r["book_id"] not in allowed_books:
            continue
        authors = json.loads(r["authors"]) if isinstance(r["authors"], str) else r["authors"]
        book = {"title": r["book_title"], "authors": authors, "year": r.get("year")}
        results.append(
            {
                "chunk_id": r["id"],
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "authors": authors,
                "year": r.get("year"),
                "page_start": r["page"],
                "page_end": r["page"],
                "text": r["caption"],
                "score": 0.5,
                "citation_ready": _make_citation(book, r["page"], r["page"]),
                "drive_web_view_link": r.get("drive_web_view_link", ""),
                "result_type": r.get("result_type", "figure"),
            }
        )
    return results
