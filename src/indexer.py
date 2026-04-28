"""
indexer.py
==========
Builds and manages an inverted index over crawled pages.

Index structure (stored as JSON):
{
  "word": {
    "df": <int>,               # document frequency
    "postings": {
      "url": {
        "tf": <int>,           # raw term frequency
        "tfidf": <float>,      # TF-IDF score
        "positions": [<int>]   # 0-based word positions in the page
      }
    }
  }
}

Supports:
- Case-insensitive indexing
- Stop-word filtering
- TF-IDF scoring (log-normalised TF × IDF)
- JSON serialisation / deserialisation
"""

import json
import math
import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common English stop words — not indexed
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "our", "their", "what", "which", "who", "whom",
    "s", "t", "re", "ve", "ll", "d", "m",
}

# Regex that keeps only alphabetic characters (strips punctuation/numbers)
_TOKEN_RE = re.compile(r"[a-z]+")


class Indexer:
    """
    Builds an inverted index from crawled page data and supports
    serialisation to / from a JSON file.
    """

    def __init__(self):
        # {word: {"df": int, "postings": {url: {"tf": int, "positions": list}}}}
        self._index: dict[str, dict[str, Any]] = {}
        self._num_docs: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def add_page(self, page: dict) -> None:
        """
        Index a single page produced by the Crawler.

        Args:
            page: dict with keys 'url' and 'text'
        """
        url = page["url"]
        tokens = self._tokenise(page.get("text", "") + " " + page.get("title", ""))
        if not tokens:
            return

        self._num_docs += 1
        word_positions: dict[str, list[int]] = {}

        for position, word in enumerate(tokens):
            word_positions.setdefault(word, []).append(position)

        for word, positions in word_positions.items():
            if word not in self._index:
                self._index[word] = {"df": 0, "postings": {}}
            entry = self._index[word]
            entry["df"] += 1
            entry["postings"][url] = {
                "tf": len(positions),
                "positions": positions,
                "tfidf": 0.0,  # calculated in _compute_tfidf
            }

        self._compute_tfidf()
        logger.debug("Indexed %s  (%d unique tokens)", url, len(word_positions))

    def build_from_pages(self, pages: list[dict]) -> None:
        """Index a list of pages (convenience wrapper)."""
        for page in pages:
            self.add_page(page)

    def get_word(self, word: str) -> dict | None:
        """
        Return the full index entry for a word, or None if not found.
        Entry format: {"df": int, "postings": {url: {"tf", "tfidf", "positions"}}}
        """
        return self._index.get(word.lower())

    def find(self, query: str) -> list[tuple[str, float]]:
        """
        Find pages containing ALL query words.
        Returns a list of (url, score) tuples sorted by TF-IDF score descending.

        Multi-word queries use intersection: only pages containing every
        query term are returned. The score is the sum of per-term TF-IDF scores.
        """
        words = [w for w in self._tokenise(query) if w]
        if not words:
            return []

        # Build sets of matching URLs per word
        posting_sets = []
        for word in words:
            entry = self._index.get(word)
            if entry is None:
                return []  # Any missing word → no results
            posting_sets.append(set(entry["postings"].keys()))

        # Intersection of all sets
        common_urls = posting_sets[0]
        for s in posting_sets[1:]:
            common_urls &= s

        # Score: sum of TF-IDF across all query terms for each URL
        scored = []
        for url in common_urls:
            score = sum(
                self._index[word]["postings"][url]["tfidf"]
                for word in words
            )
            scored.append((url, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def vocab_size(self) -> int:
        return len(self._index)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the index to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_meta": {"num_docs": self._num_docs, "vocab_size": len(self._index)},
            "index": self._index,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Index saved to %s  (%d words, %d docs)", path, len(self._index), self._num_docs)

    def load(self, path: str | Path) -> None:
        """Load the index from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._index = payload["index"]
        self._num_docs = payload["_meta"]["num_docs"]
        logger.info("Index loaded from %s  (%d words, %d docs)", path, len(self._index), self._num_docs)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> list[str]:
        """
        Lowercase, extract alphabetic tokens, remove stop words.
        Returns a list of tokens in document order (positions preserved).
        """
        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def _compute_tfidf(self) -> None:
        """
        Recompute TF-IDF scores for every posting.

        TF  = 1 + log10(raw_tf)        (log-normalised)
        IDF = log10(N / df)            (standard IDF)
        TF-IDF = TF × IDF
        """
        N = self._num_docs
        if N == 0:
            return
        for word, entry in self._index.items():
            df = entry["df"]
            idf = math.log10(N / df) if df > 0 else 0.0
            for url, posting in entry["postings"].items():
                tf = 1 + math.log10(posting["tf"]) if posting["tf"] > 0 else 0.0
                posting["tfidf"] = round(tf * idf, 4)
# Added stop-word filtering
