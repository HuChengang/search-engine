"""
crawler.py
==========
Web crawler for quotes.toscrape.com.

Responsibilities:
- Discover all page URLs starting from a seed URL
- Download and parse each page's HTML
- Respect a politeness window between requests
- Return structured page data for the indexer
"""

import time
import logging
from urllib.parse import urljoin, urlparse
from typing import Generator

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Crawler:
    """
    Breadth-first web crawler with politeness window enforcement.

    Attributes:
        seed_url      : Starting URL for the crawl
        politeness    : Minimum seconds to wait between HTTP requests
        session       : Persistent requests.Session for connection reuse
        visited       : Set of already-visited URLs (deduplication)
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; XJCO3011-SearchBot/1.0; "
            "+https://github.com/HuChengang/search-engine)"
        )
    }

    def __init__(self, seed_url: str = "https://quotes.toscrape.com/", politeness: float = 6.0):
        self.seed_url = seed_url.rstrip("/")
        self.politeness = politeness
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self.visited: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def crawl(self) -> Generator[dict, None, None]:
        """
        Crawl the entire site breadth-first.

        Yields dicts with keys:
            url     : Canonical page URL
            title   : <title> text or empty string
            text    : Visible body text (newline-separated)
            links   : List of absolute same-domain URLs found on the page
        """
        queue = [self.seed_url]
        self.visited.clear()

        while queue:
            url = queue.pop(0)
            if url in self.visited:
                continue

            page = self._fetch(url)
            if page is None:
                continue

            self.visited.add(url)
            new_links = [
                link for link in page["links"]
                if link not in self.visited and link not in queue
            ]
            queue.extend(new_links)

            logger.info("Crawled: %s  (queue: %d)", url, len(queue))
            yield page

            # Politeness window — always wait before the next request
            if queue:
                logger.debug("Waiting %.1f s (politeness window)...", self.politeness)
                time.sleep(self.politeness)

    def fetch_page(self, url: str) -> dict | None:
        """Public single-page fetch (used by tests)."""
        return self._fetch(url)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch(self, url: str) -> dict | None:
        """
        Download a single URL, parse it, and return structured data.
        Returns None on any network or HTTP error.
        """
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        return {
            "url": url,
            "title": self._extract_title(soup),
            "text": self._extract_text(soup),
            "links": self._extract_links(soup, url),
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract visible text from the page body.
        Removes script/style tags before extraction.
        """
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        body = soup.find("body")
        if body is None:
            return ""
        lines = [line.strip() for line in body.get_text(separator="\n").splitlines()]
        return "\n".join(line for line in lines if line)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """
        Extract same-domain absolute URLs from all <a href> tags.
        Strips fragments and query strings for deduplication.
        """
        base_domain = urlparse(self.seed_url).netloc
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            # Only follow links on the same domain, HTTP/HTTPS only
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != base_domain:
                continue
            # Normalise: drop fragment, keep path + query
            clean = parsed._replace(fragment="").geturl()
            if clean not in links:
                links.append(clean)
        return links
# Crawler complete
# Fixed: politeness window now enforced correctly
# Added User-Agent header
# Added edge case: empty body pages
