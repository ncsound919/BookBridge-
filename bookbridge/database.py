"""SQLite database layer for BookBridge."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import DB_PATH


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables and FTS indices if they don't already exist."""
    conn = _connect(db_path)
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                drive_file_id TEXT,
                drive_account TEXT DEFAULT '',
                title TEXT NOT NULL,
                authors TEXT NOT NULL DEFAULT '[]',
                year INTEGER,
                publisher TEXT DEFAULT '',
                isbn TEXT DEFAULT '',
                doi TEXT DEFAULT '',
                subject_areas TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                mime_type TEXT DEFAULT '',
                drive_web_view_link TEXT DEFAULT '',
                last_indexed_at TEXT,
                pages INTEGER DEFAULT 0,
                hash TEXT DEFAULT '',
                language TEXT DEFAULT 'en',
                edition TEXT DEFAULT '',
                abstract TEXT DEFAULT '',
                ocr_applied INTEGER DEFAULT 0,
                equations_indexed INTEGER DEFAULT 0,
                figures_indexed INTEGER DEFAULT 0,
                allowed_agents TEXT NOT NULL DEFAULT '[]',
                local_path TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS book_chunks (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_start INTEGER NOT NULL DEFAULT 0,
                page_end INTEGER NOT NULL DEFAULT 0,
                section_heading TEXT DEFAULT '',
                embedding BLOB
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS book_chunks_fts USING fts5(
                text,
                section_heading,
                content='book_chunks',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS equations (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                page INTEGER NOT NULL DEFAULT 0,
                latex TEXT DEFAULT '',
                rendered_text TEXT NOT NULL DEFAULT '',
                section_heading TEXT DEFAULT '',
                embedding BLOB
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS equations_fts USING fts5(
                rendered_text,
                latex,
                section_heading,
                content='equations',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS figures (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                page INTEGER NOT NULL DEFAULT 0,
                figure_number TEXT DEFAULT '',
                caption TEXT NOT NULL DEFAULT '',
                result_type TEXT NOT NULL DEFAULT 'figure',
                embedding BLOB
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS figures_fts USING fts5(
                caption,
                figure_number,
                content='figures',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                from_type TEXT NOT NULL,
                to_id TEXT NOT NULL,
                to_type TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_graph_from ON graph_edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_graph_to ON graph_edges(to_id);

            CREATE TABLE IF NOT EXISTS activity_references (
                id TEXT PRIMARY KEY,
                activity_type TEXT NOT NULL,
                activity_id TEXT NOT NULL,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                page_start INTEGER,
                page_end INTEGER,
                reason TEXT DEFAULT '',
                agent_or_process_id TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                page INTEGER NOT NULL DEFAULT 0,
                highlight_text TEXT DEFAULT '',
                note TEXT DEFAULT '',
                color TEXT DEFAULT '',
                source TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                book_ids TEXT DEFAULT '[]',
                progress_json TEXT DEFAULT '{}'
            );
            """
        )
    conn.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _json(v: Any) -> str:
    return json.dumps(v)


def _load(v: str) -> Any:
    if v is None:
        return []
    try:
        return json.loads(v)
    except Exception:
        return v


# ── books ─────────────────────────────────────────────────────────────────────

def upsert_book(conn: sqlite3.Connection, book: dict) -> str:
    """Insert or replace a book record. Returns book id."""
    book_id = book.get("id") or _new_id()
    conn.execute(
        """
        INSERT INTO books (
            id, drive_file_id, drive_account, title, authors, year,
            publisher, isbn, doi, subject_areas, tags, mime_type,
            drive_web_view_link, last_indexed_at, pages, hash, language,
            edition, abstract, ocr_applied, equations_indexed, figures_indexed,
            allowed_agents, local_path
        ) VALUES (
            :id, :drive_file_id, :drive_account, :title, :authors, :year,
            :publisher, :isbn, :doi, :subject_areas, :tags, :mime_type,
            :drive_web_view_link, :last_indexed_at, :pages, :hash, :language,
            :edition, :abstract, :ocr_applied, :equations_indexed,
            :figures_indexed, :allowed_agents, :local_path
        )
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            authors=excluded.authors,
            year=excluded.year,
            publisher=excluded.publisher,
            isbn=excluded.isbn,
            doi=excluded.doi,
            subject_areas=excluded.subject_areas,
            tags=excluded.tags,
            mime_type=excluded.mime_type,
            drive_web_view_link=excluded.drive_web_view_link,
            last_indexed_at=excluded.last_indexed_at,
            pages=excluded.pages,
            hash=excluded.hash,
            language=excluded.language,
            edition=excluded.edition,
            abstract=excluded.abstract,
            ocr_applied=excluded.ocr_applied,
            equations_indexed=excluded.equations_indexed,
            figures_indexed=excluded.figures_indexed,
            allowed_agents=excluded.allowed_agents,
            local_path=excluded.local_path
        """,
        {
            "id": book_id,
            "drive_file_id": book.get("drive_file_id", ""),
            "drive_account": book.get("drive_account", ""),
            "title": book.get("title", ""),
            "authors": _json(book.get("authors", [])),
            "year": book.get("year"),
            "publisher": book.get("publisher", ""),
            "isbn": book.get("isbn", ""),
            "doi": book.get("doi", ""),
            "subject_areas": _json(book.get("subject_areas", [])),
            "tags": _json(book.get("tags", [])),
            "mime_type": book.get("mime_type", ""),
            "drive_web_view_link": book.get("drive_web_view_link", ""),
            "last_indexed_at": book.get("last_indexed_at", _now()),
            "pages": book.get("pages", 0),
            "hash": book.get("hash", ""),
            "language": book.get("language", "en"),
            "edition": book.get("edition", ""),
            "abstract": book.get("abstract", ""),
            "ocr_applied": int(book.get("ocr_applied", False)),
            "equations_indexed": book.get("equations_indexed", 0),
            "figures_indexed": book.get("figures_indexed", 0),
            "allowed_agents": _json(book.get("allowed_agents", [])),
            "local_path": book.get("local_path", ""),
        },
    )
    return book_id


def get_book(conn: sqlite3.Connection, book_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    if row is None:
        return None
    return _row_to_book(row)


def _row_to_book(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("authors", "subject_areas", "tags", "allowed_agents"):
        d[key] = _load(d.get(key, "[]"))
    return d


def list_books(
    conn: sqlite3.Connection,
    filter_tag: Optional[str] = None,
    filter_author: Optional[str] = None,
    filter_year_from: Optional[int] = None,
    filter_year_to: Optional[int] = None,
    filter_subject_area: Optional[str] = None,
) -> list[dict]:
    rows = conn.execute("SELECT * FROM books").fetchall()
    results = [_row_to_book(r) for r in rows]

    if filter_tag:
        results = [b for b in results if filter_tag in b.get("tags", [])]
    if filter_author:
        low = filter_author.lower()
        results = [
            b
            for b in results
            if any(low in a.lower() for a in b.get("authors", []))
        ]
    if filter_year_from is not None:
        results = [b for b in results if (b.get("year") or 0) >= filter_year_from]
    if filter_year_to is not None:
        results = [b for b in results if (b.get("year") or 9999) <= filter_year_to]
    if filter_subject_area:
        low = filter_subject_area.lower()
        results = [
            b
            for b in results
            if any(low in s.lower() for s in b.get("subject_areas", []))
        ]
    return results


def delete_book_chunks(conn: sqlite3.Connection, book_id: str) -> None:
    # Keep external-content FTS index in sync with book_chunks.
    # Delete FTS rows whose rowid matches rows being removed from book_chunks.
    with conn:
        conn.execute(
            """
            DELETE FROM book_chunks_fts
            WHERE rowid IN (
                SELECT rowid FROM book_chunks WHERE book_id=?
            )
            """,
            (book_id,),
        )
        conn.execute("DELETE FROM book_chunks WHERE book_id=?", (book_id,))


# ── chunks ────────────────────────────────────────────────────────────────────

def insert_chunk(
    conn: sqlite3.Connection,
    book_id: str,
    chunk_index: int,
    text: str,
    page_start: int,
    page_end: int,
    section_heading: str = "",
    embedding: Optional[bytes] = None,
) -> str:
    chunk_id = _new_id()
    conn.execute(
        """
        INSERT INTO book_chunks
            (id, book_id, chunk_index, text, page_start, page_end, section_heading, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, book_id, chunk_index, text, page_start, page_end, section_heading, embedding),
    )
    # Keep FTS in sync
    conn.execute(
        "INSERT INTO book_chunks_fts(rowid, text, section_heading) VALUES (last_insert_rowid(), ?, ?)",
        (text, section_heading),
    )
    return chunk_id


def get_chunks_for_book(
    conn: sqlite3.Connection, book_id: str, page_start: int, page_end: int
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM book_chunks
        WHERE book_id=? AND page_start <= ? AND page_end >= ?
        ORDER BY chunk_index
        """,
        (book_id, page_end, page_start),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_chunks_with_embeddings(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, book_id, text, page_start, page_end, section_heading, embedding FROM book_chunks WHERE embedding IS NOT NULL"
    ).fetchall()
    return [dict(r) for r in rows]


# ── equations ─────────────────────────────────────────────────────────────────

def insert_equation(
    conn: sqlite3.Connection,
    book_id: str,
    page: int,
    rendered_text: str,
    latex: str = "",
    section_heading: str = "",
    embedding: Optional[bytes] = None,
) -> str:
    eq_id = _new_id()
    conn.execute(
        """
        INSERT INTO equations
            (id, book_id, page, latex, rendered_text, section_heading, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (eq_id, book_id, page, latex, rendered_text, section_heading, embedding),
    )
    conn.execute(
        "INSERT INTO equations_fts(rowid, rendered_text, latex, section_heading) VALUES (last_insert_rowid(), ?, ?, ?)",
        (rendered_text, latex, section_heading),
    )
    return eq_id


def search_equations_fts(conn: sqlite3.Connection, query: str, book_ids: Optional[list] = None, max_results: int = 5) -> list[dict]:
    sql = """
        SELECT e.*, b.title as book_title, b.authors, b.year, b.drive_web_view_link
        FROM equations e
        JOIN books b ON e.book_id = b.id
        WHERE e.rowid IN (SELECT rowid FROM equations_fts WHERE equations_fts MATCH ?)
    """
    params: list = [query]
    if book_ids:
        placeholders = ",".join("?" * len(book_ids))
        sql += f" AND e.book_id IN ({placeholders})"
        params.extend(book_ids)
    sql += " LIMIT ?"
    params.append(max_results)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ── figures ───────────────────────────────────────────────────────────────────

def insert_figure(
    conn: sqlite3.Connection,
    book_id: str,
    page: int,
    figure_number: str,
    caption: str,
    result_type: str = "figure",
    embedding: Optional[bytes] = None,
) -> str:
    fig_id = _new_id()
    conn.execute(
        """
        INSERT INTO figures
            (id, book_id, page, figure_number, caption, result_type, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (fig_id, book_id, page, figure_number, caption, result_type, embedding),
    )
    conn.execute(
        "INSERT INTO figures_fts(rowid, caption, figure_number) VALUES (last_insert_rowid(), ?, ?)",
        (caption, figure_number),
    )
    return fig_id


def search_figures_fts(
    conn: sqlite3.Connection,
    query: str,
    result_type: str = "both",
    book_ids: Optional[list] = None,
    max_results: int = 5,
) -> list[dict]:
    sql = """
        SELECT f.*, b.title as book_title, b.authors, b.year, b.drive_web_view_link
        FROM figures f
        JOIN books b ON f.book_id = b.id
        WHERE f.rowid IN (SELECT rowid FROM figures_fts WHERE figures_fts MATCH ?)
    """
    params: list = [query]
    if result_type != "both":
        sql += " AND f.result_type = ?"
        params.append(result_type)
    if book_ids:
        placeholders = ",".join("?" * len(book_ids))
        sql += f" AND f.book_id IN ({placeholders})"
        params.extend(book_ids)
    sql += " LIMIT ?"
    params.append(max_results)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ── graph edges ───────────────────────────────────────────────────────────────

def insert_graph_edge(
    conn: sqlite3.Connection,
    from_id: str,
    from_type: str,
    to_id: str,
    to_type: str,
    relation: str,
    weight: float = 1.0,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO graph_edges
            (id, from_id, from_type, to_id, to_type, relation, weight, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (_new_id(), from_id, from_type, to_id, to_type, relation, weight, _now()),
    )


def get_related_nodes(
    conn: sqlite3.Connection,
    seed_id: str,
    max_hops: int = 2,
    max_results: int = 10,
) -> tuple[list[dict], list[dict]]:
    """BFS over graph_edges up to max_hops. Returns (nodes, edges)."""
    visited_ids: set = set()
    frontier = {seed_id}
    all_edges: list[dict] = []

    for _ in range(max_hops):
        if not frontier:
            break
        placeholders = ",".join("?" * len(frontier))
        rows = conn.execute(
            f"""
            SELECT * FROM graph_edges
            WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})
            """,
            list(frontier) * 2,
        ).fetchall()
        new_frontier: set = set()
        for r in rows:
            d = dict(r)
            all_edges.append(d)
            for nid in (d["from_id"], d["to_id"]):
                if nid not in visited_ids:
                    new_frontier.add(nid)
        visited_ids.update(frontier)
        frontier = new_frontier - visited_ids

    # Deduplicate edges
    seen_edge_ids: set = set()
    unique_edges = []
    for e in all_edges:
        if e["id"] not in seen_edge_ids:
            seen_edge_ids.add(e["id"])
            unique_edges.append(e)

    # Build node list from edge endpoints
    all_node_ids = {seed_id}
    for e in unique_edges:
        all_node_ids.update([e["from_id"], e["to_id"]])

    # Attempt to resolve book nodes
    nodes = []
    for nid in list(all_node_ids)[:max_results]:
        book = get_book(conn, nid)
        if book:
            nodes.append(
                {
                    "id": nid,
                    "type": "book",
                    "label": book["title"],
                    "book_title": book["title"],
                    "page": None,
                    "relevance_score": 1.0,
                }
            )
        else:
            nodes.append(
                {
                    "id": nid,
                    "type": "concept",
                    "label": nid,
                    "book_title": None,
                    "page": None,
                    "relevance_score": 1.0,
                }
            )

    return nodes, unique_edges


# ── activity references ───────────────────────────────────────────────────────

def insert_activity_reference(
    conn: sqlite3.Connection,
    activity_type: str,
    activity_id: str,
    book_id: str,
    page_start: Optional[int],
    page_end: Optional[int],
    reason: str = "",
    agent_or_process_id: str = "",
) -> str:
    ref_id = _new_id()
    conn.execute(
        """
        INSERT INTO activity_references
            (id, activity_type, activity_id, book_id, page_start, page_end,
             reason, agent_or_process_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ref_id,
            activity_type,
            activity_id,
            book_id,
            page_start,
            page_end,
            reason,
            agent_or_process_id,
            _now(),
        ),
    )
    return ref_id


# ── index jobs ────────────────────────────────────────────────────────────────

def create_index_job(conn: sqlite3.Connection, book_ids: list) -> str:
    job_id = _new_id()
    now = _now()
    conn.execute(
        """
        INSERT INTO index_jobs (id, status, created_at, updated_at, book_ids, progress_json)
        VALUES (?, 'queued', ?, ?, ?, '{}')
        """,
        (job_id, now, now, _json(book_ids)),
    )
    return job_id


def update_index_job(conn: sqlite3.Connection, job_id: str, status: str, progress: dict = {}) -> None:
    conn.execute(
        "UPDATE index_jobs SET status=?, updated_at=?, progress_json=? WHERE id=?",
        (status, _now(), _json(progress), job_id),
    )


def get_index_job(conn: sqlite3.Connection, job_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM index_jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["book_ids"] = _load(d.get("book_ids", "[]"))
    d["progress_json"] = _load(d.get("progress_json", "{}"))
    return d


# ── stats ─────────────────────────────────────────────────────────────────────

def get_stats(conn: sqlite3.Connection) -> dict:
    books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM book_chunks").fetchone()[0]
    equations = conn.execute("SELECT COUNT(*) FROM equations").fetchone()[0]
    figures = conn.execute("SELECT COUNT(*) FROM figures").fetchone()[0]
    return {
        "books_indexed": books,
        "chunks_indexed": chunks,
        "equations_indexed": equations,
        "figures_indexed": figures,
    }
