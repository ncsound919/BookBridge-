"""BookBridge HTTP daemon (port 8777).

Start with: python -m bookbridge.server
or via the entry point: bookbridge-server

The MCP server is implemented in ``bookbridge.mcp_server`` and is started
via its own entry point."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .citation import format_citation
from .config import DB_PATH, HTTP_HOST, HTTP_PORT
from .database import (
    _connect,
    create_annotation,
    create_index_job,
    get_book,
    get_chunks_for_book,
    get_index_job,
    get_related_nodes,
    get_stats,
    init_db,
    insert_activity_reference,
    list_annotations,
    list_books,
    search_equations_fts,
    search_figures_fts,
    update_index_job,
    upsert_book,
)
from .indexer import index_book, load_cached_book
from .search import search as do_search
from .summarize import generate_flashcards, summarize_content

# ── global DB connection state (thread-local connections, WAL mode) ─────────

_db_state = threading.local()
_db_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    """
    Return a SQLite connection for the current thread.

    Each thread gets its own connection instance to avoid concurrent use of a
    single sqlite3.Connection across threads, which is not safe even with
    check_same_thread=False.
    """
    if not hasattr(_db_state, "conn") or _db_state.conn is None:
        _db_state.conn = _connect(DB_PATH)
    return _db_state.conn


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    get_db()  # prime the connection for the main thread
    yield
    # Close the main-thread connection if it was created.
    if hasattr(_db_state, "conn") and _db_state.conn is not None:
        _db_state.conn.close()
        _db_state.conn = None


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BookBridge",
    version=__version__,
    description="Local daemon providing agents with seamless access to a book library.",
    lifespan=lifespan,
)

# ── Pydantic models ───────────────────────────────────────────────────────────


class SearchFilters(BaseModel):
    book_ids: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    year_range: Optional[list[int]] = Field(None, min_length=2, max_length=2)
    authors: Optional[list[str]] = None
    subject_areas: Optional[list[str]] = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(10, ge=1, le=50)
    min_score: float = 0.0
    search_mode: str = Field("hybrid", pattern="^(hybrid|semantic|keyword)$")
    include_equations: bool = False
    include_figures: bool = False
    filters: Optional[SearchFilters] = None
    include_context_chunks: bool = False
    context_window_chunks: int = 2
    agent_or_process_id: Optional[str] = None


class RetrieveRequest(BaseModel):
    book_id: str
    page_start: Optional[int] = Field(None, ge=1)
    page_end: Optional[int] = Field(None, ge=1)
    chunk_id: Optional[str] = None


class EquationsRequest(BaseModel):
    query: str
    max_results: int = Field(5, ge=1)
    book_ids: Optional[list[str]] = None


class FiguresRequest(BaseModel):
    query: str
    result_type: str = Field("both", pattern="^(figure|table|both)$")
    max_results: int = Field(5, ge=1)
    book_ids: Optional[list[str]] = None


class RelatedRequest(BaseModel):
    seed: dict  # {"type": "concept|book_id|chunk_id", "value": "..."}
    max_hops: int = Field(2, ge=1, le=4)
    max_results: int = Field(10, ge=1)


class ReadingPlanRequest(BaseModel):
    topic: str
    goal: Optional[str] = None
    max_books: int = Field(5, ge=1)
    max_passages_per_book: int = Field(3, ge=1)


class CitationRequest(BaseModel):
    book_id: str
    style: str = Field(..., pattern="^(APA|MLA|Chicago|BibTeX|Vancouver|IEEE)$")
    page_start: Optional[int] = Field(None, ge=1)
    page_end: Optional[int] = Field(None, ge=1)
    chapter: Optional[str] = None


class ActivityReference(BaseModel):
    book_id: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    reason: Optional[str] = ""


class LinkActivityRequest(BaseModel):
    activity_type: str
    activity_id: str
    agent_or_process_id: Optional[str] = ""
    references: list[ActivityReference]


class IndexRequest(BaseModel):
    book_ids: Optional[list[str]] = None
    force: bool = False
    ocr_fallback: bool = True


class AddBookRequest(BaseModel):
    local_path: str
    title: Optional[str] = None
    authors: Optional[list[str]] = []
    year: Optional[int] = None
    publisher: Optional[str] = ""
    isbn: Optional[str] = ""
    doi: Optional[str] = ""
    subject_areas: Optional[list[str]] = []
    tags: Optional[list[str]] = []
    allowed_agents: Optional[list[str]] = []


class CreateAnnotationRequest(BaseModel):
    book_id: str
    page: int = Field(..., ge=1)
    highlight_text: Optional[str] = ""
    note: Optional[str] = ""
    color: Optional[str] = ""
    source: Optional[str] = ""


class SummarizeRequest(BaseModel):
    book_id: str
    page_start: int = Field(1, ge=1)
    page_end: int = Field(1, ge=1)
    max_sentences: int = Field(5, ge=1, le=20)
    query: Optional[str] = None


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    conn = get_db()
    stats = get_stats(conn)
    return {
        "version": __version__,
        "drive_connected": False,
        "offline_mode": True,
        "accounts": 0,
        **stats,
        "last_indexed_at": None,
    }


@app.post("/books/add", status_code=201)
def add_book(req: AddBookRequest):
    """Add a local file to the index. Returns book_id."""
    path = Path(req.local_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.local_path}")
    conn = get_db()
    meta = req.model_dump(exclude={"local_path"})
    meta["local_path"] = str(path)
    with _db_lock:
        book_id = index_book(conn, path, meta)
    return {"book_id": book_id}


@app.post("/index", status_code=202)
def trigger_index(req: IndexRequest):
    conn = get_db()
    with conn:
        job_id = create_index_job(conn, req.book_ids or [])
    # Run indexing asynchronously in a background thread
    threading.Thread(
        target=_run_index_job, args=(job_id, req.book_ids, req.force), daemon=True
    ).start()
    return {"job_id": job_id, "status": "queued"}


def _run_index_job(job_id: str, book_ids: Optional[list], force: bool) -> None:
    conn = get_db()
    with conn:
        update_index_job(conn, job_id, "indexing")
    # In this implementation, indexing is triggered via /books/add.
    # This endpoint is a placeholder for Drive-based re-indexing.
    with conn:
        update_index_job(conn, job_id, "done")


@app.get("/index/progress")
def index_progress(job_id: str = Query(...)):
    conn = get_db()
    job = get_index_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_generator():
        yield f"data: {json.dumps(job)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/search")
def search_books(req: SearchRequest):
    conn = get_db()
    filters = req.filters.model_dump() if req.filters else None
    result = do_search(
        conn,
        query=req.query,
        max_results=req.max_results,
        search_mode=req.search_mode,
        include_equations=req.include_equations,
        include_figures=req.include_figures,
        filters=filters,
        agent_id=req.agent_or_process_id,
    )
    # Apply min_score filter
    result["results"] = [r for r in result["results"] if r.get("score", 0) >= req.min_score]
    return result


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    conn = get_db()
    book = get_book(conn, req.book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    page_start = req.page_start or 1
    page_end = req.page_end or page_start

    # Try DB chunks first
    chunks = get_chunks_for_book(conn, req.book_id, page_start, page_end)
    if chunks:
        text = "\n\n".join(c["text"] for c in chunks)
    else:
        # Fall back to offline cache
        pages = load_cached_book(req.book_id)
        if pages is None:
            raise HTTPException(status_code=404, detail="Content not available (not indexed)")
        selected = pages[max(0, page_start - 1) : page_end]
        text = "\n\n".join(selected)

    def stream():
        chunk_size = 1024
        for i in range(0, len(text), chunk_size):
            yield f"data: {json.dumps({'text': text[i:i+chunk_size]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/equations")
def equations(req: EquationsRequest):
    conn = get_db()
    try:
        rows = search_equations_fts(
            conn,
            req.query,
            book_ids=req.book_ids,
            max_results=req.max_results,
        )
    except sqlite3.OperationalError as exc:
        # Malformed FTS5 query (e.g., bad quotes/operators) should return a 400, not a 500.
        raise HTTPException(
            status_code=400,
            detail=f"Invalid equation search query: {exc}",
        )
    results = []
    for r in rows:
        book = get_book(conn, r["book_id"])
        if book is None:
            continue
        results.append(
            {
                "equation_id": r["id"],
                "book_id": r["book_id"],
                "book_title": book["title"],
                "page": r["page"],
                "latex": r.get("latex") or None,
                "rendered_text": r["rendered_text"],
                "section_heading": r.get("section_heading", ""),
                "citation_ready": format_citation(book, "APA", r["page"], r["page"])[0],
            }
        )
    return {"results": results}


@app.post("/figures")
def figures(req: FiguresRequest):
    conn = get_db()
    try:
        rows = search_figures_fts(
            conn,
            req.query,
            result_type=req.result_type,
            book_ids=req.book_ids,
            max_results=req.max_results,
        )
    except sqlite3.OperationalError as exc:
        # Handle malformed FTS5 queries gracefully instead of returning 500
        raise HTTPException(status_code=400, detail=f"Malformed search query: {exc}") from exc
    results = []
    for r in rows:
        book = get_book(conn, r["book_id"])
        if book is None:
            continue
        results.append(
            {
                "figure_id": r["id"],
                "book_id": r["book_id"],
                "book_title": book["title"],
                "page": r["page"],
                "figure_number": r.get("figure_number", ""),
                "caption": r["caption"],
                "result_type": r.get("result_type", "figure"),
                "citation_ready": format_citation(book, "APA", r["page"], r["page"])[0],
                "drive_web_view_link": book.get("drive_web_view_link", ""),
            }
        )
    return {"results": results}


@app.post("/graph/related")
def graph_related(req: RelatedRequest):
    conn = get_db()
    seed_value = req.seed.get("value", "")
    nodes, edges = get_related_nodes(conn, seed_value, req.max_hops, req.max_results)
    edge_list = [
        {
            "from": e["from_id"],
            "to": e["to_id"],
            "relation": e["relation"],
        }
        for e in edges
    ]
    return {"nodes": nodes, "edges": edge_list}


@app.post("/reading_plan")
def reading_plan(req: ReadingPlanRequest):
    """Generate a reading plan by searching the topic and ordering by relevance."""
    conn = get_db()
    result = do_search(conn, req.topic, max_results=req.max_books * req.max_passages_per_book)

    # Group by book
    books_seen: dict[str, dict] = {}
    for item in result["results"]:
        bid = item["book_id"]
        if bid not in books_seen:
            books_seen[bid] = {
                "book_id": bid,
                "book_title": item["book_title"],
                "authors": item["authors"],
                "passages": [],
                "top_score": item["score"],
            }
        if len(books_seen[bid]["passages"]) < req.max_passages_per_book:
            books_seen[bid]["passages"].append(
                {
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "summary": item["text"][:200],
                    "citation_ready": item["citation_ready"],
                }
            )

    sorted_books = sorted(books_seen.values(), key=lambda b: b["top_score"], reverse=True)[
        : req.max_books
    ]

    plan = []
    for i, b in enumerate(sorted_books, start=1):
        book_full = get_book(conn, b["book_id"])
        plan.append(
            {
                "step": i,
                "book_id": b["book_id"],
                "book_title": b["book_title"],
                "authors": b["authors"],
                "rationale": f"Ranked #{i} by relevance to '{req.topic}'.",
                "passages": b["passages"],
            }
        )

    return {"plan": plan, "topic": req.topic, "goal": req.goal or ""}


@app.post("/citation")
def citation(req: CitationRequest):
    conn = get_db()
    book = get_book(conn, req.book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    cite_str, bibtex_key = format_citation(
        book, req.style, req.page_start, req.page_end, req.chapter
    )
    return {"citation": cite_str, "bibtex_key": bibtex_key}


@app.post("/link_activity", status_code=201)
def link_activity(req: LinkActivityRequest):
    conn = get_db()
    for ref in req.references:
        book = get_book(conn, ref.book_id)
        if book is None:
            raise HTTPException(status_code=400, detail=f"Book not found: {ref.book_id}")
        with conn:
            insert_activity_reference(
                conn,
                activity_type=req.activity_type,
                activity_id=req.activity_id,
                book_id=ref.book_id,
                page_start=ref.page_start,
                page_end=ref.page_end,
                reason=ref.reason or "",
                agent_or_process_id=req.agent_or_process_id or "",
            )
    return {"status": "created"}


@app.get("/books")
def books_list(
    tag: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
    subject_area: Optional[str] = Query(None),
):
    conn = get_db()
    books = list_books(
        conn,
        filter_tag=tag,
        filter_author=author,
        filter_year_from=year_from,
        filter_year_to=year_to,
        filter_subject_area=subject_area,
    )
    return [
        {
            "book_id": b["id"],
            "title": b["title"],
            "authors": b["authors"],
            "year": b.get("year"),
            "drive_web_view_link": b.get("drive_web_view_link", ""),
        }
        for b in books
    ]


@app.get("/books/{book_id}")
def book_detail(book_id: str):
    conn = get_db()
    book = get_book(conn, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@app.get("/annotations")
def annotations(
    book_id: Optional[str] = Query(None),
    page_start: Optional[int] = Query(None),
    page_end: Optional[int] = Query(None),
):
    conn = get_db()
    results = list_annotations(conn, book_id=book_id, page_start=page_start, page_end=page_end)
    return results


@app.post("/annotations", status_code=201)
def create_annotation_endpoint(req: CreateAnnotationRequest):
    conn = get_db()
    book = get_book(conn, req.book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    with conn:
        ann_id = create_annotation(
            conn,
            book_id=req.book_id,
            page=req.page,
            highlight_text=req.highlight_text or "",
            note=req.note or "",
            color=req.color or "",
            source=req.source or "",
        )
    return {"annotation_id": ann_id}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    conn = get_db()
    result = summarize_content(
        conn,
        book_id=req.book_id,
        page_start=req.page_start,
        page_end=req.page_end,
        max_sentences=req.max_sentences,
        query=req.query,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/books/{book_id}/flashcards")
def flashcards(book_id: str, max_cards: int = Query(20, ge=1, le=100)):
    conn = get_db()
    result = generate_flashcards(conn, book_id=book_id, max_cards=max_cards)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn

    uvicorn.run(app, host=HTTP_HOST, port=HTTP_PORT)


if __name__ == "__main__":
    main()
