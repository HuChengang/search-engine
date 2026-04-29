"""
test_search.py
==============
Unit tests for the SearchEngine (search.py) module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from indexer import Indexer
from search import SearchEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    idx = Indexer()
    idx.add_page({
        "url": "https://quotes.toscrape.com/",
        "title": "Quotes",
        "text": "Life is what happens when you're busy making other plans.",
    })
    idx.add_page({
        "url": "https://quotes.toscrape.com/page/2/",
        "title": "Page Two",
        "text": "Get busy living or get busy dying.",
    })
    return SearchEngine(idx)


# ── cmd_print ─────────────────────────────────────────────────────────────────

class TestCmdPrint:
    def test_known_word_shows_df(self, engine):
        output = engine.cmd_print("busy")
        assert "Document Frequency" in output
        assert "busy" in output.lower()

    def test_known_word_shows_url(self, engine):
        output = engine.cmd_print("busy")
        assert "quotes.toscrape.com" in output

    def test_known_word_shows_tf(self, engine):
        output = engine.cmd_print("busy")
        assert "Term Frequency" in output

    def test_known_word_shows_tfidf(self, engine):
        output = engine.cmd_print("busy")
        assert "TF-IDF" in output

    def test_known_word_shows_positions(self, engine):
        output = engine.cmd_print("busy")
        assert "Positions" in output

    def test_unknown_word_returns_not_found(self, engine):
        output = engine.cmd_print("xyzzy")
        assert "not found" in output.lower()

    def test_empty_string_returns_error(self, engine):
        output = engine.cmd_print("")
        assert "error" in output.lower() or "please" in output.lower()

    def test_case_insensitive(self, engine):
        lower = engine.cmd_print("busy")
        upper = engine.cmd_print("BUSY")
        # Both should find something (same word)
        assert "not found" not in lower.lower()
        assert "not found" not in upper.lower()

    def test_whitespace_only_returns_error(self, engine):
        output = engine.cmd_print("   ")
        assert "error" in output.lower() or "not found" in output.lower()


# ── cmd_find ──────────────────────────────────────────────────────────────────

class TestCmdFind:
    def test_single_word_finds_page(self, engine):
        output = engine.cmd_find("life")
        assert "quotes.toscrape.com" in output

    def test_result_shows_rank(self, engine):
        output = engine.cmd_find("busy")
        assert "#1" in output

    def test_result_shows_score(self, engine):
        output = engine.cmd_find("busy")
        assert "Score" in output

    def test_result_shows_count(self, engine):
        output = engine.cmd_find("busy")
        assert "page(s)" in output

    def test_multi_word_intersection(self, engine):
        # "busy" is in pages but "xyzzy" is never indexed → no intersection
        output = engine.cmd_find("busy xyzzy")
        assert "No pages found" in output

    def test_multi_word_same_page(self, engine):
        # Both "busy" and "living" are on page 2
        output = engine.cmd_find("busy living")
        assert "page/2/" in output

    def test_unknown_word_returns_no_results(self, engine):
        output = engine.cmd_find("xyzzy")
        assert "No pages found" in output

    def test_empty_query_returns_error(self, engine):
        output = engine.cmd_find("")
        assert "error" in output.lower() or "please" in output.lower()

    def test_stop_words_only_returns_no_results(self, engine):
        output = engine.cmd_find("the and or")
        assert "No pages found" in output

    def test_case_insensitive(self, engine):
        lower = engine.cmd_find("life")
        upper = engine.cmd_find("LIFE")
        # Compare results ignoring the query string display
        assert "quotes.toscrape.com" in lower
        assert "quotes.toscrape.com" in upper

    def test_output_contains_query(self, engine):
        output = engine.cmd_find("busy")
        assert "busy" in output.lower()
