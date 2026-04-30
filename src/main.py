"""
main.py
=======
Command-line interface for the SpotifyInsights Search Engine.

Commands
--------
  build           Crawl https://quotes.toscrape.com/ and build the index
  load            Load a previously built index from disk
  print <word>    Print the inverted index entry for <word>
  find <query>    Find pages containing all words in <query>
  help            Show this help message
  quit / exit     Exit the shell

Usage
-----
  python main.py            # start interactive shell
  python main.py build      # run a single command and exit
"""

import sys
import logging
from pathlib import Path

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).parent))

from crawler import Crawler
from indexer import Indexer
from search import SearchEngine

# ── Configuration ─────────────────────────────────────────────────────────────

SEED_URL    = "https://quotes.toscrape.com/"
POLITENESS  = 6.0          # seconds between HTTP requests
INDEX_PATH  = Path(__file__).parent.parent / "data" / "index.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── ANSI colours (gracefully degrade on Windows) ──────────────────────────────
try:
    import colorama
    colorama.init()
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"
except ImportError:
    GREEN = YELLOW = RED = CYAN = RESET = ""

BANNER = f"""
{GREEN}╔══════════════════════════════════════════╗
║       🔍  Search Engine  v1.0            ║
║       XJCO3011 Coursework 2              ║
╚══════════════════════════════════════════╝{RESET}
Type {CYAN}help{RESET} for available commands.
"""

HELP_TEXT = f"""
{CYAN}Available commands:{RESET}
  {GREEN}build{RESET}          Crawl {SEED_URL} and build the inverted index
  {GREEN}load{RESET}           Load the index from {INDEX_PATH}
  {GREEN}print{RESET} <word>   Print the inverted index entry for <word>
  {GREEN}find{RESET} <query>   Find pages containing all words in <query>
  {GREEN}help{RESET}           Show this message
  {GREEN}quit{RESET}           Exit the shell
"""


# ── Shell ─────────────────────────────────────────────────────────────────────

class Shell:
    def __init__(self):
        self.indexer = Indexer()
        self.engine  = SearchEngine(self.indexer)
        self._loaded = False

    def run(self):
        print(BANNER)
        while True:
            try:
                raw = input(f"{GREEN}>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd   = parts[0].lower()
            args  = parts[1] if len(parts) > 1 else ""

            self._dispatch(cmd, args)

    def run_once(self, raw: str):
        """Execute a single command string (used by CLI argument mode)."""
        parts = raw.strip().split(maxsplit=1)
        cmd   = parts[0].lower()
        args  = parts[1] if len(parts) > 1 else ""
        self._dispatch(cmd, args)

    # ── Command dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, cmd: str, args: str):
        if cmd == "build":
            self._cmd_build()
        elif cmd == "load":
            self._cmd_load()
        elif cmd == "print":
            self._require_index()
            print(self.engine.cmd_print(args))
        elif cmd == "find":
            self._require_index()
            print(self.engine.cmd_find(args))
        elif cmd in ("help", "?"):
            print(HELP_TEXT)
        elif cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            sys.exit(0)
        else:
            print(f"{RED}Unknown command: '{cmd}'.{RESET}  Type {CYAN}help{RESET} for available commands.")

    # ── Individual command implementations ────────────────────────────────────

    def _cmd_build(self):
        print(f"{YELLOW}Building index…{RESET}")
        print(f"  Seed URL   : {SEED_URL}")
        print(f"  Politeness : {POLITENESS}s between requests")
        print(f"  Index path : {INDEX_PATH}\n")

        crawler = Crawler(seed_url=SEED_URL, politeness=POLITENESS)
        self.indexer = Indexer()
        pages_crawled = 0

        for page in crawler.crawl():
            self.indexer.add_page(page)
            pages_crawled += 1
            print(f"  [{pages_crawled:03d}] Indexed: {page['url']}")

        self.indexer.save(INDEX_PATH)
        self.engine = SearchEngine(self.indexer)
        self._loaded = True

        print(f"\n{GREEN}✓ Build complete!{RESET}")
        print(f"  Pages crawled : {pages_crawled}")
        print(f"  Vocabulary    : {self.indexer.vocab_size} unique words")
        print(f"  Saved to      : {INDEX_PATH}")

    def _cmd_load(self):
        if not INDEX_PATH.exists():
            print(f"{RED}Error:{RESET} Index file not found at {INDEX_PATH}.")
            print(f"  Run {CYAN}build{RESET} first to create it.")
            return
        print(f"{YELLOW}Loading index from {INDEX_PATH}…{RESET}")
        self.indexer = Indexer()
        self.indexer.load(INDEX_PATH)
        self.engine  = SearchEngine(self.indexer)
        self._loaded = True
        print(f"{GREEN}✓ Index loaded!{RESET}")
        print(f"  Documents : {self.indexer.num_docs}")
        print(f"  Vocabulary: {self.indexer.vocab_size} unique words")

    def _require_index(self):
        if not self._loaded:
            print(f"{YELLOW}Warning:{RESET} No index loaded. Run {CYAN}build{RESET} or {CYAN}load{RESET} first.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    shell = Shell()
    if len(sys.argv) > 1:
        # Non-interactive mode: python main.py "find good friends"
        shell.run_once(" ".join(sys.argv[1:]))
    else:
        shell.run()
# Added ANSI colour support
# Added single-command mode
