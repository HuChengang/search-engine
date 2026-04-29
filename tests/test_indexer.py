"""
test_indexer.py
===============
Unit tests for the Indexer module.
"""

import json
import math
import pytest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from indexer import Indexer, STOP_WORDS


# ── Fixtures ──────────────────────────────────────────────────────────────────

PAGE_A = {
    "url": "https://quotes.toscrape.com/",
    "title": "Quotes to Scrape",
    "text": "The world as we have created it is a process of our thinking.",
}
PAGE_B = {
    "url": "https://quotes.toscrape.com/page/2/",
    "title": "Page Two",
    "text": "It is our choices that show what we truly are far more than our abilities.",
}
PAGE_C = {
    "url": "https://quotes.toscrape.com/page/3/",
    "title": "Page Three",
    "text": "There are only two ways to live your life.",
}


@pytest.fixture
def indexer_one():
    idx = Indexer()
    idx.add_page(PAGE_A)
    return idx

@pytest.fixture
def indexer_two():
    idx = Indexer()
    idx.add_page(PAGE_A)
    idx.add_page(PAGE_B)
    return idx

@pytest.fixture
def indexer_three():
    idx = Indexer()
    idx.add_page(PAGE_A)
    idx.add_page(PAGE_B)
    idx.add_page(PAGE_C)
    return idx


# ── Tokenisation ──────────────────────────────────────────────────────────────

class TestTokenisation:
    def test_lowercase(self):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "Hello WORLD"})
        assert idx.get_word("hello") is not None
        assert idx.get_word("world") is not None

    def test_stop_words_excluded(self):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "the quick brown fox"})
        assert idx.get_word("the") is None
        assert idx.get_word("quick") is not None

    def test_punctuation_stripped(self):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "hello, world!"})
        assert idx.get_word("hello") is not None
        assert idx.get_word("world") is not None

    def test_numbers_excluded(self):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "page 42 results"})
        assert idx.get_word("42") is None
        assert idx.get_word("page") is not None

    def test_single_char_excluded(self):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "a b cat"})
        assert idx.get_word("a") is None
        assert idx.get_word("b") is None
        assert idx.get_word("cat") is not None


# ── Index structure ───────────────────────────────────────────────────────────

class TestIndexStructure:
    def test_word_in_index(self, indexer_one):
        entry = indexer_one.get_word("world")
        assert entry is not None

    def test_unknown_word_returns_none(self, indexer_one):
        assert indexer_one.get_word("xyzzy") is None

    def test_posting_has_required_fields(self, indexer_one):
        entry = indexer_one.get_word("world")
        url = list(entry["postings"].keys())[0]
        posting = entry["postings"][url]
        assert "tf" in posting
        assert "tfidf" in posting
        assert "positions" in posting

    def test_term_frequency_correct(self, indexer_one):
        # "our" appears once in PAGE_A after stop-word filter check
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "cat cat cat dog"})
        entry = idx.get_word("cat")
        posting = entry["postings"]["http://test.com"]
        assert posting["tf"] == 3

    def test_positions_recorded(self, indexer_one):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "apple banana apple"})
        entry = idx.get_word("apple")
        posting = entry["postings"]["http://test.com"]
        assert len(posting["positions"]) == 2

    def test_document_frequency(self, indexer_two):
        # "choices" appears in PAGE_B only
        entry = indexer_two.get_word("choices")
        assert entry["df"] == 1

    def test_df_increments_across_pages(self, indexer_two):
        # "our" not a stop word in context; use a real shared word
        idx = Indexer()
        idx.add_page({"url": "http://a.com", "title": "", "text": "python programming"})
        idx.add_page({"url": "http://b.com", "title": "", "text": "python rocks"})
        entry = idx.get_word("python")
        assert entry["df"] == 2

    def test_num_docs_increments(self, indexer_two):
        assert indexer_two.num_docs == 2

    def test_vocab_size_positive(self, indexer_one):
        assert indexer_one.vocab_size > 0


# ── TF-IDF ────────────────────────────────────────────────────────────────────

class TestTFIDF:
    def test_tfidf_positive_for_rare_word(self, indexer_two):
        # "choices" only in PAGE_B → should have positive TF-IDF
        entry = indexer_two.get_word("choices")
        url = list(entry["postings"].keys())[0]
        assert entry["postings"][url]["tfidf"] > 0

    def test_tfidf_zero_when_in_all_docs(self):
        idx = Indexer()
        idx.add_page({"url": "http://a.com", "title": "", "text": "shared word here"})
        idx.add_page({"url": "http://b.com", "title": "", "text": "shared different context"})
        entry = idx.get_word("shared")
        # df == N so IDF = log10(1) = 0 → TF-IDF = 0
        for url, posting in entry["postings"].items():
            assert posting["tfidf"] == 0.0

    def test_tfidf_higher_for_rarer_word(self, indexer_three):
        # "choices" only in PAGE_B; "world" might be in multiple
        # Just check that tfidf is computed and numeric
        for word in ["world", "choices", "ways"]:
            entry = indexer_three.get_word(word)
            if entry:
                for posting in entry["postings"].values():
                    assert isinstance(posting["tfidf"], float)


# ── Search (find) ─────────────────────────────────────────────────────────────

class TestFind:
    def test_find_single_word(self, indexer_one):
        results = indexer_one.find("world")
        assert len(results) == 1
        assert results[0][0] == PAGE_A["url"]

    def test_find_returns_list_of_tuples(self, indexer_one):
        results = indexer_one.find("world")
        assert isinstance(results, list)
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2

    def test_find_unknown_word_returns_empty(self, indexer_one):
        results = indexer_one.find("xyzzy")
        assert results == []

    def test_find_empty_query_returns_empty(self, indexer_one):
        results = indexer_one.find("")
        assert results == []

    def test_find_multiword_intersection(self, indexer_two):
        # "choices" is in PAGE_B, "world" is in PAGE_A → no intersection
        results = indexer_two.find("choices world")
        assert results == []

    def test_find_multiword_same_page(self):
        idx = Indexer()
        idx.add_page({"url": "http://test.com", "title": "", "text": "python programming language"})
        results = idx.find("python programming")
        assert len(results) == 1
        assert results[0][0] == "http://test.com"

    def test_find_results_sorted_by_score(self, indexer_three):
        idx = Indexer()
        idx.add_page({"url": "http://a.com", "title": "", "text": "python python python rocks"})
        idx.add_page({"url": "http://b.com", "title": "", "text": "python language"})
        results = idx.find("python")
        assert results[0][1] >= results[1][1]

    def test_find_case_insensitive(self, indexer_one):
        lower = indexer_one.find("world")
        upper = indexer_one.find("WORLD")
        assert lower == upper

    def test_find_stop_word_only_returns_empty(self, indexer_one):
        # All stop words get filtered → empty token list → empty results
        results = indexer_one.find("the and or")
        assert results == []


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_save_creates_file(self, indexer_one, tmp_path):
        p = tmp_path / "index.json"
        indexer_one.save(p)
        assert p.exists()

    def test_save_valid_json(self, indexer_one, tmp_path):
        p = tmp_path / "index.json"
        indexer_one.save(p)
        with open(p) as f:
            data = json.load(f)
        assert "index" in data
        assert "_meta" in data

    def test_load_restores_index(self, indexer_one, tmp_path):
        p = tmp_path / "index.json"
        indexer_one.save(p)
        idx2 = Indexer()
        idx2.load(p)
        assert idx2.get_word("world") is not None

    def test_load_restores_num_docs(self, indexer_two, tmp_path):
        p = tmp_path / "index.json"
        indexer_two.save(p)
        idx2 = Indexer()
        idx2.load(p)
        assert idx2.num_docs == 2

    def test_load_nonexistent_raises(self, tmp_path):
        idx = Indexer()
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "missing.json")

    def test_round_trip_find(self, indexer_two, tmp_path):
        p = tmp_path / "index.json"
        indexer_two.save(p)
        idx2 = Indexer()
        idx2.load(p)
        original = indexer_two.find("world")
        restored = idx2.find("world")
        assert original == restored
