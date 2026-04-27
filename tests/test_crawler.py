"""
test_crawler.py
===============
Unit and integration tests for the Crawler module.
Uses unittest.mock to avoid real HTTP requests.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from crawler import Crawler


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
  <p>Hello world, this is a test page.</p>
  <a href="/page/2/">Next</a>
  <a href="https://external.com/page">External</a>
  <a href="/page/2/">Duplicate</a>
</body>
</html>
"""

EMPTY_HTML = "<html><body></body></html>"

def make_mock_response(text: str, status_code: int = 200):
    mock = MagicMock()
    mock.text = text
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        mock.raise_for_status.side_effect = requests.HTTPError(response=mock)
    return mock


# ── Crawler initialisation ────────────────────────────────────────────────────

class TestCrawlerInit:
    def test_default_seed_url(self):
        c = Crawler()
        assert c.seed_url == "https://quotes.toscrape.com"

    def test_custom_seed_url(self):
        c = Crawler(seed_url="https://example.com/")
        assert c.seed_url == "https://example.com"

    def test_default_politeness(self):
        c = Crawler()
        assert c.politeness == 6.0

    def test_custom_politeness(self):
        c = Crawler(politeness=1.0)
        assert c.politeness == 1.0

    def test_visited_starts_empty(self):
        c = Crawler()
        assert len(c.visited) == 0


# ── Link extraction ───────────────────────────────────────────────────────────

class TestLinkExtraction:
    def setup_method(self):
        self.crawler = Crawler(seed_url="https://quotes.toscrape.com/", politeness=0)

    @patch("crawler.requests.Session.get")
    def test_extracts_same_domain_links(self, mock_get):
        mock_get.return_value = make_mock_response(SAMPLE_HTML)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page/2/" in result["links"]

    @patch("crawler.requests.Session.get")
    def test_excludes_external_links(self, mock_get):
        mock_get.return_value = make_mock_response(SAMPLE_HTML)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert all("external.com" not in link for link in result["links"])

    @patch("crawler.requests.Session.get")
    def test_deduplicates_links(self, mock_get):
        mock_get.return_value = make_mock_response(SAMPLE_HTML)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        page2_links = [l for l in result["links"] if "/page/2/" in l]
        assert len(page2_links) == 1


# ── Text extraction ───────────────────────────────────────────────────────────

class TestTextExtraction:
    def setup_method(self):
        self.crawler = Crawler(seed_url="https://quotes.toscrape.com/", politeness=0)

    @patch("crawler.requests.Session.get")
    def test_extracts_body_text(self, mock_get):
        mock_get.return_value = make_mock_response(SAMPLE_HTML)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert "Hello world" in result["text"]

    @patch("crawler.requests.Session.get")
    def test_extracts_title(self, mock_get):
        mock_get.return_value = make_mock_response(SAMPLE_HTML)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert result["title"] == "Test Page"

    @patch("crawler.requests.Session.get")
    def test_empty_body_returns_empty_text(self, mock_get):
        mock_get.return_value = make_mock_response(EMPTY_HTML)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert result["text"] == ""

    @patch("crawler.requests.Session.get")
    def test_strips_script_tags(self, mock_get):
        html = "<html><body><script>var x=1;</script><p>Clean</p></body></html>"
        mock_get.return_value = make_mock_response(html)
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert "var x" not in result["text"]
        assert "Clean" in result["text"]


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def setup_method(self):
        self.crawler = Crawler(seed_url="https://quotes.toscrape.com/", politeness=0)

    @patch("crawler.requests.Session.get")
    def test_returns_none_on_http_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.HTTPError("404")
        result = self.crawler.fetch_page("https://quotes.toscrape.com/bad")
        assert result is None

    @patch("crawler.requests.Session.get")
    def test_returns_none_on_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("No connection")
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert result is None

    @patch("crawler.requests.Session.get")
    def test_returns_none_on_timeout(self, mock_get):
        import requests as req
        mock_get.side_effect = req.Timeout("Timeout")
        result = self.crawler.fetch_page("https://quotes.toscrape.com/")
        assert result is None


# ── Crawl deduplication ───────────────────────────────────────────────────────

class TestCrawlDeduplication:
    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session.get")
    def test_does_not_revisit_urls(self, mock_get, mock_sleep):
        html_with_self_link = """
        <html><head><title>T</title></head>
        <body><p>Text</p><a href="/">Home</a></body></html>
        """
        mock_get.return_value = make_mock_response(html_with_self_link)
        crawler = Crawler(seed_url="https://quotes.toscrape.com/", politeness=0)
        pages = list(crawler.crawl())
        # Should only crawl root once despite self-link
        urls = [p["url"] for p in pages]
        assert len(urls) == len(set(urls))

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session.get")
    def test_politeness_sleep_called(self, mock_get, mock_sleep):
        html_with_link = """
        <html><head><title>T</title></head>
        <body><a href="/page/2/">Next</a></body></html>
        """
        html_page2 = "<html><head><title>P2</title></head><body><p>End</p></body></html>"
        mock_get.side_effect = [
            make_mock_response(html_with_link),
            make_mock_response(html_page2),
        ]
        crawler = Crawler(seed_url="https://quotes.toscrape.com/", politeness=6.0)
        list(crawler.crawl())
        mock_sleep.assert_called_with(6.0)
