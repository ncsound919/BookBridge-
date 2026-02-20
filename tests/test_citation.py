"""Tests for citation formatter."""

import pytest

from bookbridge.citation import format_citation


SAMPLE_BOOK = {
    "title": "Introduction to Quantum Mechanics",
    "authors": ["David J. Griffiths"],
    "year": 2005,
    "publisher": "Pearson Prentice Hall",
    "isbn": "978-0-13-191175-7",
    "doi": "10.0000/iqm",
    "edition": "2nd",
}

MULTI_AUTHOR_BOOK = {
    "title": "Machine Learning",
    "authors": ["Tom Mitchell", "Yoshua Bengio", "Geoffrey Hinton"],
    "year": 1997,
    "publisher": "McGraw-Hill",
}


def test_apa_single_author():
    cite, bibtex = format_citation(SAMPLE_BOOK, "APA", page_start=42, page_end=45)
    assert "Griffiths" in cite
    assert "2005" in cite
    assert "Quantum Mechanics" in cite
    assert bibtex is None


def test_apa_page_range():
    cite, _ = format_citation(SAMPLE_BOOK, "APA", page_start=10, page_end=20)
    assert "10" in cite
    assert "20" in cite


def test_apa_single_page():
    cite, _ = format_citation(SAMPLE_BOOK, "APA", page_start=7, page_end=7)
    assert "p. 7" in cite


def test_mla():
    cite, bibtex = format_citation(SAMPLE_BOOK, "MLA")
    assert "Griffiths" in cite
    assert "Quantum Mechanics" in cite
    assert bibtex is None


def test_chicago():
    cite, _ = format_citation(SAMPLE_BOOK, "Chicago")
    assert "Griffiths" in cite
    assert "2005" in cite


def test_bibtex_single_author():
    cite, key = format_citation(SAMPLE_BOOK, "BibTeX")
    assert "@book" in cite
    assert "griffiths2005" == key
    assert "Griffiths" in cite
    assert "Quantum Mechanics" in cite


def test_bibtex_multiple_authors():
    cite, key = format_citation(MULTI_AUTHOR_BOOK, "BibTeX")
    assert "mitchell1997" == key
    assert "and" in cite  # authors joined with " and "


def test_vancouver():
    cite, _ = format_citation(SAMPLE_BOOK, "Vancouver")
    assert "Griffiths" in cite
    assert "2005" in cite


def test_ieee():
    cite, _ = format_citation(SAMPLE_BOOK, "IEEE")
    assert "Griffiths" in cite
    assert "Quantum Mechanics" in cite


def test_no_authors():
    book = {"title": "Unknown Book", "authors": [], "year": 2000}
    cite, _ = format_citation(book, "APA")
    assert "Unknown Author" in cite


def test_no_year():
    book = {"title": "A Book", "authors": ["A. Author"], "year": None}
    cite, _ = format_citation(book, "APA")
    assert "n.d." in cite


def test_case_insensitive_style():
    cite1, _ = format_citation(SAMPLE_BOOK, "APA")
    cite2, _ = format_citation(SAMPLE_BOOK, "apa")
    assert cite1 == cite2
