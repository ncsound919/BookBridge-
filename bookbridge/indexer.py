"""Book indexer: extract text from PDF/EPUB/plain-text files and store chunks."""

from __future__ import annotations

import gzip
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Optional

from .config import CACHE_DIR, CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from .database import (
    delete_book_chunks,
    insert_chunk,
    insert_equation,
    insert_figure,
    upsert_book,
)
from .embedder import get_embedder

# ── text extraction ───────────────────────────────────────────────────────────


def _extract_pdf(path: Path) -> tuple[list[str], int]:
    """Return (pages_text_list, page_count). Each element is one page's text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return pages, len(pages)
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed for {path}: {exc}") from exc


def _extract_epub(path: Path) -> tuple[list[str], int]:
    """Return (pages_text_list, page_count). EPUBs have no physical pages;
    each spine item becomes a pseudo-page."""
    try:
        import ebooklib
        from ebooklib import epub
        import html

        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        pages = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            raw = item.get_content().decode("utf-8", errors="replace")
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", raw)
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append(text)
        return pages, len(pages)
    except Exception as exc:
        raise RuntimeError(f"EPUB extraction failed for {path}: {exc}") from exc


def _extract_text(path: Path) -> tuple[list[str], int]:
    """Treat plain text as one big page."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [text], 1


def extract_text_from_file(path: Path) -> tuple[list[str], int]:
    """Dispatch to the right extractor based on file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".epub":
        return _extract_epub(path)
    # Fallback: plain text
    return _extract_text(path)


# ── chunking ──────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── equation / figure detection ───────────────────────────────────────────────

_EQUATION_RE = re.compile(
    r"(\\\[.+?\\\]|\\\(.+?\\\)|"            # LaTeX display/inline
    r"\$\$.+?\$\$|\$.+?\$|"                  # $...$ and $$...$$
    r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[^\n.;]{3,60})",  # simple assignments
    re.DOTALL,
)

_FIGURE_RE = re.compile(
    r"(?:Figure|Fig\.?|Table)\s+(\d+[A-Za-z]?)[.:]\s*([^\n]{10,300})",
    re.IGNORECASE,
)


def _detect_equations(text: str, page: int) -> list[dict]:
    results = []
    for m in _EQUATION_RE.finditer(text):
        eq_text = m.group(0).strip()
        # Decide whether it looks like LaTeX
        latex = eq_text if re.search(r"[\\$]", eq_text) else ""
        results.append({"page": page, "latex": latex, "rendered_text": eq_text})
    return results


def _detect_figures(text: str, page: int) -> list[dict]:
    results = []
    for m in _FIGURE_RE.finditer(text):
        keyword = m.group(0).lower()
        result_type = "table" if keyword.startswith("table") else "figure"
        results.append(
            {
                "page": page,
                "figure_number": m.group(1),
                "caption": m.group(2).strip(),
                "result_type": result_type,
            }
        )
    return results


# ── section heading detection ─────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^(?:\d+\.?\d*\.?\s+)?([A-Z][A-Za-z0-9 :,\-]{3,80})\s*$",
    re.MULTILINE,
)


def _extract_headings(text: str) -> list[str]:
    return [m.group(1).strip() for m in _HEADING_RE.finditer(text)]


def _nearest_heading(text: str) -> str:
    headings = _extract_headings(text)
    return headings[0] if headings else ""


# ── file hash ─────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── offline cache ─────────────────────────────────────────────────────────────

def cache_book(book_id: str, pages: list[str]) -> None:
    cache_path = CACHE_DIR / f"{book_id}.txt.gz"
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        f.write("\n\n--- PAGE BREAK ---\n\n".join(pages))


def load_cached_book(book_id: str) -> Optional[list[str]]:
    cache_path = CACHE_DIR / f"{book_id}.txt.gz"
    if not cache_path.exists():
        return None
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        return f.read().split("\n\n--- PAGE BREAK ---\n\n")


# ── main indexer ──────────────────────────────────────────────────────────────


def index_book(
    conn: sqlite3.Connection,
    file_path: Path,
    book_meta: Optional[dict] = None,
    progress_cb=None,
) -> str:
    """Index a book file and store all artefacts in *conn*.

    Returns the book_id.

    *progress_cb* is called with (pages_done, pages_total) as each page is processed.
    """
    if book_meta is None:
        book_meta = {}

    file_path = Path(file_path)
    pages, total_pages = extract_text_from_file(file_path)
    file_hash = _file_hash(file_path)

    # Build book record
    book_record = {
        "title": book_meta.get("title") or file_path.stem,
        "authors": book_meta.get("authors", []),
        "year": book_meta.get("year"),
        "publisher": book_meta.get("publisher", ""),
        "isbn": book_meta.get("isbn", ""),
        "doi": book_meta.get("doi", ""),
        "subject_areas": book_meta.get("subject_areas", []),
        "tags": book_meta.get("tags", []),
        "mime_type": book_meta.get("mime_type", ""),
        "drive_web_view_link": book_meta.get("drive_web_view_link", ""),
        "pages": total_pages,
        "hash": file_hash,
        "language": book_meta.get("language", "en"),
        "edition": book_meta.get("edition", ""),
        "abstract": book_meta.get("abstract", ""),
        "local_path": str(file_path),
        "allowed_agents": book_meta.get("allowed_agents", []),
    }
    if book_meta.get("id"):
        book_record["id"] = book_meta["id"]

    with conn:
        book_id = upsert_book(conn, book_record)
        # Remove stale chunks/equations/figures before re-indexing
        delete_book_chunks(conn, book_id)
        conn.execute("DELETE FROM equations WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM figures WHERE book_id=?", (book_id,))

    embedder = get_embedder()

    all_texts = [p for p in pages if p.strip()]
    embedder.fit(all_texts)

    equations_count = 0
    figures_count = 0
    full_text_pages: list[str] = []
    chunk_index = 0

    for page_num, page_text in enumerate(pages, start=1):
        full_text_pages.append(page_text)

        if not page_text.strip():
            if progress_cb:
                progress_cb(page_num, total_pages)
            continue

        # Extract equations and figures from this page
        eqs = _detect_equations(page_text, page_num)
        figs = _detect_figures(page_text, page_num)

        with conn:
            for eq in eqs:
                emb_vec = embedder.embed(eq["rendered_text"])
                insert_equation(
                    conn,
                    book_id=book_id,
                    page=eq["page"],
                    rendered_text=eq["rendered_text"],
                    latex=eq.get("latex", ""),
                    section_heading=_nearest_heading(page_text),
                    embedding=embedder.to_bytes(emb_vec),
                )
                equations_count += 1

            for fig in figs:
                emb_vec = embedder.embed(fig["caption"])
                insert_figure(
                    conn,
                    book_id=book_id,
                    page=fig["page"],
                    figure_number=fig["figure_number"],
                    caption=fig["caption"],
                    result_type=fig["result_type"],
                    embedding=embedder.to_bytes(emb_vec),
                )
                figures_count += 1

            # Chunk and embed the page text
            heading = _nearest_heading(page_text)
            for chunk in _chunk_text(page_text):
                if not chunk.strip():
                    continue
                emb_vec = embedder.embed(chunk)
                insert_chunk(
                    conn,
                    book_id=book_id,
                    chunk_index=chunk_index,
                    text=chunk,
                    page_start=page_num,
                    page_end=page_num,
                    section_heading=heading,
                    embedding=embedder.to_bytes(emb_vec),
                )
                chunk_index += 1

        if progress_cb:
            progress_cb(page_num, total_pages)

    # Update counts on book record
    with conn:
        conn.execute(
            "UPDATE books SET equations_indexed=?, figures_indexed=? WHERE id=?",
            (equations_count, figures_count, book_id),
        )

    # Cache for offline access
    cache_book(book_id, full_text_pages)

    return book_id
