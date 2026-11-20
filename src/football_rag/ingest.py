"""Step 1 of RAG: get the raw text.

Downloads plain-text versions of the Wikipedia pages listed in data/pages.txt
and saves each one as JSON in data/raw/. Wikipedia text is licensed CC BY-SA.
"""

import json
import re
import time
from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API users to identify themselves with a descriptive User-Agent.
USER_AGENT = "football-rag/0.1 (learning project; https://github.com/rahull97/football-rag)"

# Sections at the end of an article that are lists of links, not useful facts.
DROP_SECTIONS = {"see also", "references", "notes", "external links", "further reading",
                 "bibliography", "sources", "citations", "footnotes"}


def read_page_titles(pages_file: Path) -> list[str]:
    """Read the list of Wikipedia article titles to download.

    The file has one title per line, written exactly as on Wikipedia
    (e.g. "2014 FIFA World Cup"). Blank lines and lines starting with "#"
    (comments) are skipped, and surrounding whitespace is removed. Titles are
    returned in file order; duplicates are not removed.

    Args:
        pages_file: Path to the titles file, normally `data/pages.txt`.

    Returns:
        The article titles, in the order they appear in the file.
    """
    titles = []
    for line in pages_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    return titles


def fetch_page(title: str, session: requests.Session) -> dict:
    """Download one Wikipedia article as cleaned plain text.

    Calls the MediaWiki API (TextExtracts) with `explaintext`, so the article
    comes back without wiki markup, tables or infoboxes, and section headings
    appear as lines like "== Heading ==". Redirects are followed, so the
    returned title is Wikipedia's canonical title, which may differ from the
    one requested (e.g. "Pele" -> "Pelé"). The text is then passed through
    `clean_text` to drop link-list sections such as References.

    Args:
        title: Article title as written on Wikipedia.
        session: A `requests.Session` with the project's User-Agent set.
            Reusing one session keeps the HTTP connection open between pages.

    Returns:
        A dict with keys "title" (canonical title), "url" (full article URL)
        and "text" (cleaned plain text). This is the "page" format used by
        `load_pages`, `chunk_pages` and `FootballRAG.build_index`.

    Raises:
        requests.HTTPError: The API returned an error status.
        requests.Timeout: No response within 30 seconds.
        ValueError: No article exists with this title.
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "inprop": "url",
        "redirects": 1,
        "titles": title,
    }
    resp = session.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"Wikipedia page not found: {title!r}")
    return {
        "title": page["title"],
        "url": page["fullurl"],
        "text": clean_text(page.get("extract", "")),
    }


def clean_text(text: str) -> str:
    """Remove noise sections from an article's plain text.

    Walks the text line by line. When a top-level heading ("== Name ==") whose
    name is in `DROP_SECTIONS` (References, See also, External links, ...) is
    found, everything is dropped until the next top-level heading that is not
    in that set. Sub-headings ("=== ... ===") inside a dropped section are
    dropped with it. Kept headings stay in the text, so chunks still show
    which section they came from. Finally, runs of three or more newlines are
    collapsed to one blank line and leading/trailing whitespace is stripped.

    Args:
        text: Plain article text as returned by the Wikipedia API, with
            headings in "== Heading ==" form.

    Returns:
        The cleaned text.
    """
    kept, skipping = [], False
    for line in text.splitlines():
        heading = re.match(r"^(=+)\s*(.*?)\s*\1$", line.strip())
        if heading:
            level, name = len(heading.group(1)), heading.group(2).lower()
            if level == 2:  # a new top-level section decides whether we keep going
                skipping = name in DROP_SECTIONS
        if not skipping:
            kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def slugify(title: str) -> str:
    """Turn an article title into a safe, lower-case file name (no extension).

    Every run of characters that are not letters, digits or "_" becomes a
    single "_", leading/trailing "_" are removed, and the result is lower-cased.
    Unicode letters are kept: "2015–16 Premier League" -> "2015_16_premier_league",
    "Pelé" -> "pelé".

    Args:
        title: The article title.

    Returns:
        The slug used as the JSON file name in `data/raw/`.
    """
    return re.sub(r"[^\w]+", "_", title, flags=re.UNICODE).strip("_").lower()


def ingest(pages_file: Path, raw_dir: Path) -> list[Path]:
    """Download every article listed in `pages_file` and save it to `raw_dir`.

    This is what `football-rag ingest` runs. For each title it calls
    `fetch_page`, writes the resulting dict to `<raw_dir>/<slug>.json`
    (named after the canonical title, via `slugify`), prints one progress line,
    and waits 0.2 s before the next request to be polite to Wikipedia.

    Existing files with the same name are overwritten. Files for titles that
    were later removed from `pages_file` are NOT deleted, so they will still be
    picked up by `load_pages`; delete `data/raw/` to start clean. The run stops
    at the first page that fails (see `fetch_page` for the exceptions), and the
    pages saved before the failure stay on disk.

    Args:
        pages_file: Titles file, normally `data/pages.txt`.
        raw_dir: Output directory, created if it doesn't exist.

    Returns:
        Paths of the JSON files written, in the order of `pages_file`.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    saved = []
    for title in read_page_titles(pages_file):
        page = fetch_page(title, session)
        path = raw_dir / f"{slugify(page['title'])}.json"
        path.write_text(json.dumps(page, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  saved {page['title']:<40} {len(page['text']):>8,} chars")
        saved.append(path)
        time.sleep(0.2)  # be polite to Wikipedia's servers
    return saved


def load_pages(raw_dir: Path) -> list[dict]:
    """Load every saved page from `raw_dir`, ready for indexing.

    Reads all `*.json` files written by `ingest`, sorted by file name so the
    order (and therefore the chunk order) is the same on every run. Makes no
    network calls.

    Args:
        raw_dir: Directory written by `ingest`, normally `data/raw/`.

    Returns:
        One dict per page with keys "title", "url" and "text". The list is
        empty if the directory is missing or contains no JSON files, which
        `football-rag index` treats as "run ingest first".
    """
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]
