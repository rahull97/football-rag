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
    titles = []
    for line in pages_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    return titles


def fetch_page(title: str, session: requests.Session) -> dict:
    """Fetch one article as plain text. Headings come back as '== Heading =='."""
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
    """Remove link-list sections and collapse extra blank lines."""
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
    return re.sub(r"[^\w]+", "_", title, flags=re.UNICODE).strip("_").lower()


def ingest(pages_file: Path, raw_dir: Path) -> list[Path]:
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
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]
