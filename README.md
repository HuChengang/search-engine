# 🔍 Search Engine

A web crawler and inverted-index search engine targeting [quotes.toscrape.com](https://quotes.toscrape.com/).

> **XJCO3011 Web Services and Web Data — Coursework 2**

---

## Features

- **Breadth-first crawler** with 6-second politeness window
- **Inverted index** with TF-IDF ranking
- **4 CLI commands**: `build`, `load`, `print`, `find`
- **Multi-word queries** using intersection
- **Case-insensitive** search with stop-word filtering
- **38 automated tests** with >85% coverage

---

## Installation

```bash
git clone https://github.com/HuChengang/search-engine.git
cd search-engine
pip install -r requirements.txt
```

---

## Usage

```bash
cd src
python main.py
```

### Commands

| Command | Description |
|---------|-------------|
| `build` | Crawl the website and build the inverted index |
| `load` | Load a previously built index from `data/index.json` |
| `print <word>` | Show the inverted index entry for a word |
| `find <query>` | Find all pages containing the query terms |
| `help` | Show available commands |
| `quit` | Exit the shell |

### Examples

```
> build
> load
> print nonsense
> find indifference
> find good friends
```

---

## Project Structure

```
search-engine/
├── src/
│   ├── crawler.py     # Web crawler (BFS, politeness window)
│   ├── indexer.py     # Inverted index with TF-IDF scoring
│   ├── search.py      # Search engine CLI interface
│   └── main.py        # Interactive shell entry point
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   └── test_search.py
├── data/
│   └── index.json     # Generated index file (after build)
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Data Structures

The inverted index is stored as a nested dictionary:

```json
{
  "word": {
    "df": 3,
    "postings": {
      "https://quotes.toscrape.com/": {
        "tf": 2,
        "tfidf": 0.1761,
        "positions": [4, 17]
      }
    }
  }
}
```

- **df** — document frequency (number of pages containing the word)
- **tf** — raw term frequency within a page
- **tfidf** — TF-IDF score: `(1 + log10(tf)) × log10(N/df)`
- **positions** — 0-based token positions for proximity analysis

<!-- final version -->
