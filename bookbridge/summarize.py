"""Extractive summarization and flashcard generation utilities."""

from __future__ import annotations

import re
import sqlite3
from typing import Optional

from .database import get_book, get_chunks_for_book
from .indexer import load_cached_book


# ── summarizer ────────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _score_sentence(sentence: str, query_words: set[str]) -> float:
    """Score a sentence by keyword overlap and length heuristics."""
    words = set(re.findall(r"\b[a-z]{3,}\b", sentence.lower()))
    overlap = len(words & query_words)
    length_bonus = min(len(sentence) / 100.0, 1.0)
    return overlap + length_bonus


def summarize_content(
    conn: sqlite3.Connection,
    book_id: str,
    page_start: int,
    page_end: int,
    max_sentences: int = 5,
    query: Optional[str] = None,
) -> dict:
    """Return an extractive summary of *page_start*–*page_end* from *book_id*.

    If *query* is provided, sentences most relevant to the query are
    preferred.  Otherwise the highest-scoring by length/position are kept.
    """
    book = get_book(conn, book_id)
    if book is None:
        return {"error": f"Book not found: {book_id}"}

    chunks = get_chunks_for_book(conn, book_id, page_start, page_end)
    if chunks:
        text = " ".join(c["text"] for c in chunks)
    else:
        pages = load_cached_book(book_id)
        if pages is None:
            return {"error": "Content not available (book not indexed)"}
        text = " ".join(pages[max(0, page_start - 1) : page_end])

    sentences = _sentence_split(text)
    if not sentences:
        return {
            "book_id": book_id,
            "book_title": book["title"],
            "page_start": page_start,
            "page_end": page_end,
            "summary": "",
            "sentence_count": 0,
        }

    query_words: set[str] = set()
    if query:
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))

    scored = [(i, _score_sentence(s, query_words), s) for i, s in enumerate(sentences)]
    # Pick top sentences; preserve original order so the summary reads naturally
    top_indices = {i for i, _, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences]}
    summary_sentences = [s for i, _, s in scored if i in top_indices]

    return {
        "book_id": book_id,
        "book_title": book["title"],
        "page_start": page_start,
        "page_end": page_end,
        "summary": " ".join(summary_sentences),
        "sentence_count": len(summary_sentences),
    }


# ── flashcard generator ───────────────────────────────────────────────────────

# Patterns that suggest a definition or fact worth turning into a flashcard
_DEFN_RE = re.compile(
    r"([A-Z][A-Za-z0-9 \-]{2,60}?)\s+(?:is defined as|refers to|is called|means|denotes)\s+([^.!?]{10,200}[.!?])",
    re.IGNORECASE,
)

_COLON_RE = re.compile(
    r"^([A-Z][A-Za-z0-9 \-]{2,50}):\s+([A-Z][^.!?\n]{10,200}[.!?])",
    re.MULTILINE,
)


def _extract_flashcards_from_text(text: str, source_label: str) -> list[dict]:
    cards: list[dict] = []
    for m in _DEFN_RE.finditer(text):
        cards.append({
            "front": m.group(1).strip(),
            "back": m.group(2).strip(),
            "source": source_label,
            "card_type": "definition",
        })
    for m in _COLON_RE.finditer(text):
        cards.append({
            "front": m.group(1).strip(),
            "back": m.group(2).strip(),
            "source": source_label,
            "card_type": "term",
        })
    return cards


def generate_flashcards(conn: sqlite3.Connection, book_id: str, max_cards: int = 20) -> dict:
    """Generate study flashcards from a book's indexed content.

    Returns a list of {front, back, source, card_type} dicts derived from
    definitions, key terms, and equations found in the book.
    """
    book = get_book(conn, book_id)
    if book is None:
        return {"error": f"Book not found: {book_id}"}

    # Deduplicate by front text as we go, to avoid holding all cards in memory.
    seen: set = set()
    unique: list[dict] = []

    # Flashcards from equations
    eq_cursor = conn.execute(
        "SELECT rendered_text, latex, section_heading, page FROM equations WHERE book_id=?",
        (book_id,),
    )
    for row in eq_cursor:
        heading = row["section_heading"] or "Equation"
        eq_text = row["latex"] or row["rendered_text"]
        if not (eq_text and len(eq_text) > 2):
            continue
        card = {
            "front": heading,
            "back": eq_text,
            "source": f"p. {row['page']}",
            "card_type": "equation",
        }
        key = card["front"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(card)
        if len(unique) >= max_cards:
            break

    # Flashcards from text chunks (definition/term patterns)
    if len(unique) < max_cards:
        chunk_cursor = conn.execute(
            "SELECT text, page_start, section_heading FROM book_chunks WHERE book_id=? ORDER BY chunk_index",
            (book_id,),
        )
        for row in chunk_cursor:
            source_label = f"p. {row['page_start']}"
            chunk_cards = _extract_flashcards_from_text(row["text"], source_label)
            for card in chunk_cards:
                key = card["front"].lower()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(card)
                if len(unique) >= max_cards:
                    break
            if len(unique) >= max_cards:
                break

    return {
        "book_id": book_id,
        "book_title": book["title"],
        "flashcards": unique,
        "total_generated": len(unique),
    }
