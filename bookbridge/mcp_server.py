"""BookBridge MCP (Model Context Protocol) server on port 8778.

Exposes BookBridge capabilities as native MCP tools so any
MCP-compatible agent can call them without manual HTTP wiring.

The server implements the JSON-RPC 2.0 MCP protocol over HTTP with
SSE (server-sent events) for streaming responses.

Start with: python -m bookbridge.mcp_server
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .citation import format_citation
from .config import DB_PATH, MCP_HOST, MCP_PORT
from .database import (
    _connect,
    create_annotation,
    get_book,
    get_related_nodes,
    get_stats,
    init_db,
    insert_activity_reference,
    list_annotations,
    list_books,
    search_equations_fts,
    search_figures_fts,
)
from .indexer import load_cached_book
from .search import search as do_search, _fts_escape
from .summarize import generate_flashcards, summarize_content

# ── DB connection ─────────────────────────────────────────────────────────────

_db_state = threading.local()


def get_db() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating one if needed."""
    if not hasattr(_db_state, "conn") or _db_state.conn is None:
        init_db(DB_PATH)
        _db_state.conn = _connect(DB_PATH)
    return _db_state.conn


# ── MCP tool definitions ──────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "bookbridge_search",
        "description": (
            "Search the user's indexed book library by natural language query. "
            "Returns ranked passages with page numbers and a ready-to-use citation string. "
            "Set include_equations or include_figures to true to also surface matching "
            "equations and figure captions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "search_mode": {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "keyword"],
                    "default": "hybrid",
                },
                "include_equations": {"type": "boolean", "default": False},
                "include_figures": {"type": "boolean", "default": False},
                "filters": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "library_labels": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "bookbridge_retrieve",
        "description": (
            "Retrieve full text of a specific page range from a book. "
            "Use after bookbridge_search when the snippet is insufficient. "
            "Works offline from local cache."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "page_start": {"type": "integer", "minimum": 1},
                "page_end": {"type": "integer", "minimum": 1},
            },
            "required": ["book_id", "page_start", "page_end"],
        },
    },
    {
        "name": "bookbridge_equations",
        "description": (
            "Search for equations and LaTeX blocks across the book library by concept name "
            "or keyword. Returns LaTeX source where available, page, and citation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Equation concept or name, e.g. 'Reynolds number'",
                },
                "max_results": {"type": "integer", "default": 5},
                "book_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "bookbridge_figures",
        "description": (
            "Search figure and table captions across the book library. "
            "Returns caption text, page, figure number, and citation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "result_type": {
                    "type": "string",
                    "enum": ["figure", "table", "both"],
                    "default": "both",
                },
                "max_results": {"type": "integer", "default": 5},
                "book_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "bookbridge_related",
        "description": (
            "Query the cross-book knowledge graph to discover books, chapters, and concepts "
            "related to a given concept, book, or chunk."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seed_type": {
                    "type": "string",
                    "enum": ["concept", "book_id", "chunk_id"],
                },
                "seed_value": {"type": "string"},
                "max_hops": {"type": "integer", "default": 2, "minimum": 1, "maximum": 4},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["seed_type", "seed_value"],
        },
    },
    {
        "name": "bookbridge_reading_plan",
        "description": (
            "Generate an ordered reading plan for a research topic. Returns a prioritized "
            "list of books and passages the agent should read before implementing, "
            "simulating, or writing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "goal": {
                    "type": "string",
                    "description": "What the agent intends to do with the knowledge",
                },
                "max_books": {"type": "integer", "default": 5},
                "max_passages_per_book": {"type": "integer", "default": 3},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "bookbridge_cite",
        "description": (
            "Generate a formatted citation for a book. Call this whenever an agent uses "
            "an equation, definition, protocol, or parameter value sourced from a book."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "style": {
                    "type": "string",
                    "enum": ["APA", "MLA", "Chicago", "BibTeX", "Vancouver", "IEEE"],
                },
                "page_start": {"type": "integer", "minimum": 1},
                "page_end": {"type": "integer", "minimum": 1},
            },
            "required": ["book_id", "style"],
        },
    },
    {
        "name": "bookbridge_link_activity",
        "description": (
            "Log that the current file, notebook cell, or experiment run was informed "
            "by specific book passages. Builds a traceable reference trail."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "activity_type": {
                    "type": "string",
                    "enum": ["file", "notebook_cell", "experiment", "simulation", "run"],
                },
                "activity_id": {"type": "string"},
                "agent_or_process_id": {"type": "string"},
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "book_id": {"type": "string"},
                            "page_start": {"type": "integer"},
                            "page_end": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["book_id"],
                    },
                },
            },
            "required": ["activity_type", "activity_id", "references"],
        },
    },
    {
        "name": "bookbridge_list_books",
        "description": "List all books in the index with metadata. Use to discover available titles before searching.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter_tag": {"type": "string"},
                "filter_author": {"type": "string"},
                "filter_year_from": {"type": "integer"},
                "filter_year_to": {"type": "integer"},
                "filter_subject_area": {"type": "string"},
            },
        },
    },
    {
        "name": "bookbridge_annotate",
        "description": (
            "Save a highlight or note on a specific page of a book. "
            "Annotations are persisted locally and can be retrieved later with bookbridge_get_annotations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "page": {"type": "integer", "minimum": 1},
                "highlight_text": {"type": "string", "description": "The highlighted passage text"},
                "note": {"type": "string", "description": "User or agent note about the passage"},
                "color": {"type": "string", "description": "Highlight color label, e.g. 'yellow'"},
                "source": {"type": "string", "description": "Who or what created the annotation"},
            },
            "required": ["book_id", "page"],
        },
    },
    {
        "name": "bookbridge_get_annotations",
        "description": "Retrieve annotations (highlights and notes) for a book, optionally filtered by page range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "page_start": {"type": "integer", "minimum": 1},
                "page_end": {"type": "integer", "minimum": 1},
            },
            "required": ["book_id"],
        },
    },
    {
        "name": "bookbridge_summarize",
        "description": (
            "Generate an extractive summary of a page range from a book. "
            "Provide an optional query to focus the summary on a specific topic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "page_start": {"type": "integer", "minimum": 1},
                "page_end": {"type": "integer", "minimum": 1},
                "max_sentences": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "query": {"type": "string", "description": "Optional topic to focus the summary on"},
            },
            "required": ["book_id", "page_start", "page_end"],
        },
    },
    {
        "name": "bookbridge_flashcards",
        "description": (
            "Generate study flashcards from a book's indexed content. "
            "Returns term/definition pairs, equation cards, and key-concept cards."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "max_cards": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": ["book_id"],
        },
    },
]

# ── tool dispatch ─────────────────────────────────────────────────────────────


def _tool_search(args: dict) -> dict:
    conn = get_db()
    return do_search(
        conn,
        query=args["query"],
        max_results=args.get("max_results", 5),
        search_mode=args.get("search_mode", "hybrid"),
        include_equations=args.get("include_equations", False),
        include_figures=args.get("include_figures", False),
        filters=args.get("filters"),
    )


def _tool_retrieve(args: dict) -> dict:
    conn = get_db()
    book = get_book(conn, args["book_id"])
    if book is None:
        return {"error": f"Book not found: {args['book_id']}"}
    page_start = args["page_start"]
    page_end = args["page_end"]
    from .database import get_chunks_for_book
    chunks = get_chunks_for_book(conn, args["book_id"], page_start, page_end)
    if chunks:
        text = "\n\n".join(c["text"] for c in chunks)
    else:
        pages = load_cached_book(args["book_id"])
        if pages is None:
            return {"error": "Content not available (book not indexed)"}
        text = "\n\n".join(pages[max(0, page_start - 1) : page_end])
    return {"book_id": args["book_id"], "page_start": page_start, "page_end": page_end, "text": text}


def _tool_equations(args: dict) -> dict:
    conn = get_db()
    # Escape the query for use in FTS5 MATCH to avoid malformed syntax issues
    raw_query = args.get("query", "")
    safe_query = _fts_escape(raw_query) if raw_query is not None else ""
    try:
        rows = search_equations_fts(
            conn,
            safe_query,
            book_ids=args.get("book_ids"),
            max_results=args.get("max_results", 5),
        )
    except sqlite3.OperationalError as e:
        # Return a structured error response instead of surfacing a raw DB error
        return {"error": f"Invalid equation search query: {e}"}
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


def _tool_figures(args: dict) -> dict:
    conn = get_db()
    # Escape the FTS query to avoid malformed MATCH expressions and handle DB errors gracefully.
    safe_query = _fts_escape(args["query"])
    try:
        rows = search_figures_fts(
            conn,
            safe_query,
            result_type=args.get("result_type", "both"),
            book_ids=args.get("book_ids"),
            max_results=args.get("max_results", 5),
        )
    except sqlite3.OperationalError as exc:
        # Return a structured response instead of letting the error propagate.
        return {"results": [], "error": "Invalid full-text search query", "details": str(exc)}
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


def _tool_related(args: dict) -> dict:
    conn = get_db()
    seed_value = args["seed_value"]
    nodes, edges = get_related_nodes(conn, seed_value, args.get("max_hops", 2), args.get("max_results", 10))
    return {
        "nodes": nodes,
        "edges": [{"from": e["from_id"], "to": e["to_id"], "relation": e["relation"]} for e in edges],
    }


def _tool_reading_plan(args: dict) -> dict:
    conn = get_db()
    topic = args["topic"]
    goal = args.get("goal", "")
    max_books = args.get("max_books", 5)
    max_passages = args.get("max_passages_per_book", 3)
    result = do_search(conn, topic, max_results=max_books * max_passages)
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
        if len(books_seen[bid]["passages"]) < max_passages:
            books_seen[bid]["passages"].append(
                {
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "summary": item["text"][:200],
                    "citation_ready": item["citation_ready"],
                }
            )
    sorted_books = sorted(books_seen.values(), key=lambda b: b["top_score"], reverse=True)[:max_books]
    plan = [
        {
            "step": i + 1,
            "book_id": b["book_id"],
            "book_title": b["book_title"],
            "authors": b["authors"],
            "rationale": f"Ranked #{i + 1} by relevance to '{topic}'.",
            "passages": b["passages"],
        }
        for i, b in enumerate(sorted_books)
    ]
    return {"plan": plan, "topic": topic, "goal": goal}


def _tool_cite(args: dict) -> dict:
    conn = get_db()
    book = get_book(conn, args["book_id"])
    if book is None:
        return {"error": f"Book not found: {args['book_id']}"}
    cite_str, bibtex_key = format_citation(
        book, args["style"], args.get("page_start"), args.get("page_end")
    )
    return {"citation": cite_str, "bibtex_key": bibtex_key}


def _tool_link_activity(args: dict) -> dict:
    conn = get_db()
    for ref in args.get("references", []):
        book = get_book(conn, ref["book_id"])
        if book is None:
            return {"error": f"Book not found: {ref['book_id']}"}
        with conn:
            insert_activity_reference(
                conn,
                activity_type=args["activity_type"],
                activity_id=args["activity_id"],
                book_id=ref["book_id"],
                page_start=ref.get("page_start"),
                page_end=ref.get("page_end"),
                reason=ref.get("reason", ""),
                agent_or_process_id=args.get("agent_or_process_id", ""),
            )
    return {"status": "created"}


def _tool_list_books(args: dict) -> dict:
    conn = get_db()
    books = list_books(
        conn,
        filter_tag=args.get("filter_tag"),
        filter_author=args.get("filter_author"),
        filter_year_from=args.get("filter_year_from"),
        filter_year_to=args.get("filter_year_to"),
        filter_subject_area=args.get("filter_subject_area"),
    )
    return {
        "books": [
            {
                "book_id": b["id"],
                "title": b["title"],
                "authors": b["authors"],
                "year": b.get("year"),
                "tags": b.get("tags", []),
                "subject_areas": b.get("subject_areas", []),
                "drive_web_view_link": b.get("drive_web_view_link", ""),
            }
            for b in books
        ]
    }


def _tool_annotate(args: dict) -> dict:
    conn = get_db()
    book = get_book(conn, args["book_id"])
    if book is None:
        return {"error": f"Book not found: {args['book_id']}"}
    with conn:
        ann_id = create_annotation(
            conn,
            book_id=args["book_id"],
            page=args["page"],
            highlight_text=args.get("highlight_text", ""),
            note=args.get("note", ""),
            color=args.get("color", ""),
            source=args.get("source", ""),
        )
    return {"annotation_id": ann_id, "status": "created"}


def _tool_get_annotations(args: dict) -> dict:
    conn = get_db()
    annotations = list_annotations(
        conn,
        book_id=args.get("book_id"),
        page_start=args.get("page_start"),
        page_end=args.get("page_end"),
    )
    return {"annotations": annotations}


def _tool_summarize(args: dict) -> dict:
    conn = get_db()
    return summarize_content(
        conn,
        book_id=args["book_id"],
        page_start=args.get("page_start", 1),
        page_end=args.get("page_end", 1),
        max_sentences=args.get("max_sentences", 5),
        query=args.get("query"),
    )


def _tool_flashcards(args: dict) -> dict:
    conn = get_db()
    return generate_flashcards(conn, book_id=args["book_id"], max_cards=args.get("max_cards", 20))


_TOOL_DISPATCH = {
    "bookbridge_search": _tool_search,
    "bookbridge_retrieve": _tool_retrieve,
    "bookbridge_equations": _tool_equations,
    "bookbridge_figures": _tool_figures,
    "bookbridge_related": _tool_related,
    "bookbridge_reading_plan": _tool_reading_plan,
    "bookbridge_cite": _tool_cite,
    "bookbridge_link_activity": _tool_link_activity,
    "bookbridge_list_books": _tool_list_books,
    "bookbridge_annotate": _tool_annotate,
    "bookbridge_get_annotations": _tool_get_annotations,
    "bookbridge_summarize": _tool_summarize,
    "bookbridge_flashcards": _tool_flashcards,
}

# ── MCP FastAPI app ───────────────────────────────────────────────────────────

mcp_app = FastAPI(title="BookBridge MCP", version=__version__)


def _mcp_error(rpc_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _mcp_result(rpc_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


@mcp_app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP JSON-RPC endpoint."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_mcp_error(None, -32700, "Parse error"), status_code=400)

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    # ── MCP protocol methods ──────────────────────────────────────────────────

    if method == "initialize":
        return JSONResponse(
            _mcp_result(
                rpc_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "BookBridge", "version": __version__},
                },
            )
        )

    if method == "tools/list":
        return JSONResponse(_mcp_result(rpc_id, {"tools": TOOLS}))

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return JSONResponse(_mcp_error(rpc_id, -32601, f"Unknown tool: {tool_name}"))
        try:
            result = handler(tool_args)
        except Exception as exc:
            return JSONResponse(_mcp_error(rpc_id, -32603, str(exc)))
        return JSONResponse(
            _mcp_result(
                rpc_id,
                {
                    "content": [
                        {"type": "text", "text": json.dumps(result, indent=2)}
                    ]
                },
            )
        )

    return JSONResponse(_mcp_error(rpc_id, -32601, f"Method not found: {method}"))


@mcp_app.get("/health")
def mcp_health():
    conn = get_db()
    stats = get_stats(conn)
    return {"version": __version__, "status": "ok", **stats}


# ── SSE endpoint for streaming MCP ───────────────────────────────────────────

@mcp_app.get("/sse")
async def sse_endpoint():
    """SSE endpoint: streams a list of available tools on connection."""
    tools_payload = json.dumps({"event": "tools", "data": TOOLS})

    async def event_stream():
        yield f"data: {tools_payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn

    uvicorn.run(mcp_app, host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    main()
