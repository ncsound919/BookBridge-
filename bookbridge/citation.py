"""Citation formatter for BookBridge."""

from __future__ import annotations

from typing import Optional


def _fmt_authors_apa(authors: list[str]) -> str:
    if not authors:
        return "Unknown Author"
    parts = []
    for a in authors:
        names = a.strip().split()
        if len(names) >= 2:
            last = names[-1]
            initials = ". ".join(n[0].upper() for n in names[:-1]) + "."
            parts.append(f"{last}, {initials}")
        else:
            parts.append(a)
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", & " + parts[-1]


def _fmt_authors_mla(authors: list[str]) -> str:
    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]}, and {authors[1]}"
    return f"{authors[0]}, et al."


def _fmt_authors_chicago(authors: list[str]) -> str:
    return _fmt_authors_mla(authors)


def _fmt_authors_ieee(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    parts = []
    for a in authors:
        names = a.strip().split()
        if len(names) >= 2:
            initials = ". ".join(n[0].upper() for n in names[:-1]) + "."
            parts.append(f"{initials} {names[-1]}")
        else:
            parts.append(a)
    return ", ".join(parts)


def format_citation(
    book: dict,
    style: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    chapter: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Return (citation_string, bibtex_key_or_None)."""
    title = book.get("title", "Unknown Title")
    authors = book.get("authors") or []
    year = book.get("year") or "n.d."
    publisher = book.get("publisher", "")
    isbn = book.get("isbn", "")
    doi = book.get("doi", "")
    edition = book.get("edition", "")

    page_part = ""
    if page_start:
        if page_end and page_end != page_start:
            page_part = f"pp. {page_start}–{page_end}"
        else:
            page_part = f"p. {page_start}"

    style = style.upper()

    if style == "APA":
        author_str = _fmt_authors_apa(authors)
        parts = [f"{author_str} ({year}). *{title}*"]
        if edition:
            parts.append(f" ({edition} ed.)")
        if publisher:
            parts.append(f". {publisher}")
        if page_part:
            parts.append(f", {page_part}")
        return "".join(parts) + ".", None

    if style == "MLA":
        author_str = _fmt_authors_mla(authors)
        parts = [f'{author_str}. *{title}*']
        if edition:
            parts.append(f", {edition} ed.")
        if publisher:
            parts.append(f". {publisher}")
        if year != "n.d.":
            parts.append(f", {year}")
        if page_part:
            parts.append(f". {page_part}")
        return "".join(parts) + ".", None

    if style == "CHICAGO":
        author_str = _fmt_authors_chicago(authors)
        parts = [f"{author_str}. *{title}*"]
        if edition:
            parts.append(f", {edition} ed.")
        if publisher:
            parts.append(f". {publisher}")
        if year != "n.d.":
            parts.append(f", {year}")
        if page_part:
            parts.append(f", {page_part}")
        return "".join(parts) + ".", None

    if style == "BIBTEX":
        first_author = authors[0].split()[-1].lower() if authors else "unknown"
        bibtex_key = f"{first_author}{year}"
        author_bibtex = " and ".join(authors)
        fields = [
            f"  author = {{{author_bibtex}}}",
            f"  title = {{{title}}}",
            f"  year = {{{year}}}",
        ]
        if publisher:
            fields.append(f"  publisher = {{{publisher}}}")
        if isbn:
            fields.append(f"  isbn = {{{isbn}}}")
        if doi:
            fields.append(f"  doi = {{{doi}}}")
        if edition:
            fields.append(f"  edition = {{{edition}}}")
        body = ",\n".join(fields)
        return f"@book{{{bibtex_key},\n{body}\n}}", bibtex_key

    if style == "VANCOUVER":
        author_str = "; ".join(authors) if authors else "Unknown"
        parts = [f"{author_str}. {title}"]
        if edition:
            parts.append(f", {edition} ed.")
        if publisher:
            parts.append(f". {publisher}")
        if year != "n.d.":
            parts.append(f"; {year}")
        if page_part:
            parts.append(f": {page_part}")
        return "".join(parts) + ".", None

    if style == "IEEE":
        author_str = _fmt_authors_ieee(authors)
        parts = [f"{author_str}, *{title}*"]
        if edition:
            parts.append(f", {edition} ed.")
        if publisher:
            parts.append(f". {publisher}")
        if year != "n.d.":
            parts.append(f", {year}")
        if page_part:
            parts.append(f", {page_part}")
        return "".join(parts) + ".", None

    # Fallback
    return f"{', '.join(authors)} ({year}). {title}.", None
