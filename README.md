# BookBridge
Book tool for agents

BookBridge is a local daemon that gives any AI agent seamless access to a personal book library. It exposes books as searchable, citeable knowledge through both a **REST HTTP API** (port 8777) and a **Model Context Protocol (MCP) server** (port 8778).

## Features

- **Hybrid search** — keyword (FTS5) + semantic (TF-IDF vector) search over indexed books
- **Full-text retrieval** — stream any page range from a book, served from an offline cache
- **Equation search** — locate LaTeX blocks and mathematical expressions by concept
- **Figure & table search** — find captions and associated data
- **Knowledge graph** — cross-book concept relationships for lateral discovery
- **Reading plans** — auto-generated prioritised reading lists for research tasks
- **Citation generation** — APA, MLA, Chicago, BibTeX, Vancouver, IEEE
- **Activity linking** — traceable provenance between agent outputs and source passages
- **MCP tools** — nine native MCP tools for direct agent integration
- **Offline cache** — gzip-compressed page snapshots; search and retrieve work without internet
- **Per-book access controls** — restrict books to specific agent IDs

## Supported file types

| Format | Notes |
|--------|-------|
| PDF | Text extraction via pypdf |
| EPUB | Text extraction via ebooklib |
| Plain text | `.txt` and similar |

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Start both servers (HTTP :8777 + MCP :8778)
python main.py

# HTTP only
python main.py --http-only

# MCP only
python main.py --mcp-only
```

## Index a book

```bash
curl -X POST http://localhost:8777/books/add \
  -H 'Content-Type: application/json' \
  -d '{
    "local_path": "/path/to/my_book.pdf",
    "title": "Physics Fundamentals",
    "authors": ["Isaac Newton"],
    "year": 1687,
    "tags": ["physics", "classic"]
  }'
```

## Search

```bash
curl -X POST http://localhost:8777/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "Newton laws of motion", "max_results": 5}'
```

## MCP integration

Connect any MCP-compatible agent (Claude Desktop, Cursor, Cline, etc.) to the MCP server at `http://127.0.0.1:8778/mcp`.

### Available MCP tools

| Tool | Description |
|------|-------------|
| `bookbridge_list_books` | Discover indexed titles and metadata |
| `bookbridge_search` | Hybrid semantic + keyword search |
| `bookbridge_retrieve` | Full text of a page range |
| `bookbridge_equations` | Search equations/LaTeX by concept |
| `bookbridge_figures` | Search figure and table captions |
| `bookbridge_related` | Cross-book knowledge graph traversal |
| `bookbridge_reading_plan` | Auto-generate a prioritised reading list |
| `bookbridge_cite` | Format a citation in any style |
| `bookbridge_link_activity` | Log book provenance on files/runs |

### Recommended agent workflow

1. Call `bookbridge_reading_plan` at the start of a research task.
2. Call `bookbridge_search` (with `include_equations`/`include_figures`) for targeted retrieval.
3. Call `bookbridge_retrieve` when a search snippet is too short.
4. Call `bookbridge_equations` to look up formulas before implementing them.
5. Call `bookbridge_related` to discover laterally relevant books.
6. Call `bookbridge_cite` and embed the result wherever book-sourced material appears.
7. Call `bookbridge_link_activity` to record provenance on every file or run.

### System prompt snippet

Add this to your agent's system prompt:

> You have access to the user's book library via BookBridge MCP tools:
> `bookbridge_reading_plan`, `bookbridge_search`, `bookbridge_retrieve`,
> `bookbridge_equations`, `bookbridge_figures`, `bookbridge_related`,
> `bookbridge_cite`, `bookbridge_link_activity`, `bookbridge_list_books`.
> At the start of any research or implementation task, call
> `bookbridge_reading_plan` to orient yourself. Always call `bookbridge_cite`
> when using book-sourced material and embed the citation in your output.
> Always call `bookbridge_link_activity` to log provenance on any file, cell,
> or run you produce.

## Configuration

All settings can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKBRIDGE_DATA_DIR` | `~/.bookbridge` | Data and cache directory |
| `BOOKBRIDGE_HTTP_HOST` | `127.0.0.1` | HTTP server host |
| `BOOKBRIDGE_HTTP_PORT` | `8777` | HTTP server port |
| `BOOKBRIDGE_MCP_HOST` | `127.0.0.1` | MCP server host |
| `BOOKBRIDGE_MCP_PORT` | `8778` | MCP server port |
| `BOOKBRIDGE_CHUNK_SIZE` | `800` | Chunk size in words |
| `BOOKBRIDGE_CHUNK_OVERLAP` | `100` | Overlap between chunks in words |
| `BOOKBRIDGE_CACHE_MAX_MB` | `2048` | Max offline cache size (MB) |

## Running tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

## Project structure

```
bookbridge/
  __init__.py        Package metadata
  config.py          Environment-driven configuration
  database.py        SQLite schema, FTS5 tables, CRUD helpers
  embedder.py        TF-IDF vector embedder (numpy, no ML deps)
  indexer.py         PDF/EPUB/text extraction, chunking, indexing
  search.py          Hybrid keyword + semantic search
  citation.py        APA / MLA / Chicago / BibTeX / Vancouver / IEEE
  server.py          FastAPI HTTP daemon (port 8777)
  mcp_server.py      MCP server over HTTP/JSON-RPC (port 8778)
main.py              Entry point (starts both servers)
requirements.txt     Python dependencies
tests/               pytest test suite
```

