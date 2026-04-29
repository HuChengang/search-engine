"""
search.py
=========
High-level search interface over the Indexer.

Provides:
- cmd_print : print inverted index entry for a word
- cmd_find  : find pages matching a query, ranked by TF-IDF
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indexer import Indexer


class SearchEngine:
    """
    Thin wrapper around Indexer that formats results for the CLI.
    """

    def __init__(self, indexer: "Indexer"):
        self.indexer = indexer

    # ── Commands ──────────────────────────────────────────────────────────────

    def cmd_print(self, word: str) -> str:
        """
        Return a human-readable string for the inverted index entry of `word`.

        Example output:
            Word: "good"  |  Document Frequency: 5

            URL: https://quotes.toscrape.com/
              Term Frequency : 3
              TF-IDF Score   : 0.2141
              Positions      : [4, 17, 42]
        """
        if not word.strip():
            return "Error: please provide a word.  Usage: print <word>"

        entry = self.indexer.get_word(word.strip())
        if entry is None:
            return f'Word "{word}" not found in the index.'

        lines = [
            f'Word: "{word.lower()}"  |  Document Frequency: {entry["df"]}\n'
        ]
        for url, posting in sorted(entry["postings"].items()):
            lines.append(f"  URL: {url}")
            lines.append(f"    Term Frequency : {posting['tf']}")
            lines.append(f"    TF-IDF Score   : {posting['tfidf']}")
            lines.append(f"    Positions      : {posting['positions'][:10]}"
                         + (" ..." if len(posting["positions"]) > 10 else ""))
            lines.append("")
        return "\n".join(lines)

    def cmd_find(self, query: str) -> str:
        """
        Return a ranked list of pages containing all query terms.

        Example output:
            Query: "good friends"
            Found 3 page(s):

            #1  Score: 0.4282
                https://quotes.toscrape.com/page/2/

            #2  Score: 0.2141
                https://quotes.toscrape.com/
        """
        if not query.strip():
            return "Error: please provide a search query.  Usage: find <query>"

        results = self.indexer.find(query.strip())

        if not results:
            return f'No pages found containing all terms in: "{query}"'

        words = query.strip().lower().split()
        lines = [f'Query: "{query.strip()}"', f"Found {len(results)} page(s):\n"]
        for rank, (url, score) in enumerate(results, start=1):
            lines.append(f"  #{rank}  Score: {score}")
            lines.append(f"      {url}\n")
        return "\n".join(lines)
