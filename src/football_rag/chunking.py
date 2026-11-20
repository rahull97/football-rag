"""Step 2 of RAG: split long articles into small pieces ("chunks").

Why chunk at all? An embedding is one vector per piece of text. A whole
article squeezed into one vector blurs every topic in it together, and it is
too long to send to the model for every question anyway. Small chunks let us
find and send only the paragraphs that matter.

This is the BASELINE strategy: fixed-size windows with overlap. It ignores
the article's structure on purpose, so we can measure how much smarter
strategies (section-aware chunks, adding the article title to each chunk)
improve the evaluation scores.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """One piece of an article: the unit that gets embedded, stored and retrieved.

    Attributes:
        id: Unique id "<article title>#<position>", e.g. "Liverpool F.C.#12".
            Used as the Chroma document id.
        title: Title of the article the chunk came from. Evaluation compares
            this against the expected page titles in `eval/questions.jsonl`.
        url: URL of the source article, shown to the user as a citation.
        text: The chunk's text. Note that the article title is NOT included in
            it, so the embedding only sees the chunk's own words.
        position: 0-based index of the chunk within its article.
    """

    id: str          # unique id, e.g. "Liverpool F.C.#12"
    title: str       # article the chunk came from
    url: str
    text: str
    position: int    # index of the chunk within its article


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into windows of at most `size` characters, with overlap.

    How a window is chosen:
      * It starts at `start` and would end at `start + size`.
      * Unless it reaches the end of the text, the end is moved back to the
        last space or newline in the second half of the window, so words are
        not cut in half. If the second half has no whitespace (a very long
        token), the window is cut at exactly `size` characters.
      * The next window starts about `overlap` characters before the previous
        end, then moves forward to just after the next space/newline so it
        doesn't begin mid-word. Because of that adjustment, the shared text
        between neighbours is usually a little less than `overlap`.

    The overlap means a fact that sits on a boundary still appears whole in at
    least one chunk (as long as it is shorter than the overlap). Each piece is
    stripped of surrounding whitespace and empty pieces are dropped.

    Args:
        text: The text to split.
        size: Maximum window length in characters.
        overlap: Approximate number of characters repeated between
            consecutive windows. Must be smaller than `size`.

    Returns:
        The pieces in order. Empty if `text` is empty or only whitespace.

    Raises:
        ValueError: If `overlap >= size` (the windows would never advance).
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            newline = text.rfind("\n", start + size // 2, end)
            cut = max(space, newline)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        # Step back by `overlap`, then forward to the next word boundary so the
        # next chunk doesn't begin halfway through a word.
        next_start = max(end - overlap, start + 1)
        boundary = min((i for i in (text.find(" ", next_start), text.find("\n", next_start))
                        if i != -1 and i <= end), default=None)
        start = boundary + 1 if boundary is not None else next_start
    return pieces


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[Chunk]:
    """Split every page into `Chunk` objects using `split_fixed`.

    Each chunk carries its article's title and URL so search results can be
    shown with a citation and checked during evaluation.

    Chunk ids are built from the title, so each title must appear only once in
    `pages`. Two pages with the same title would produce duplicate ids, which
    Chroma rejects when they are added together.

    Args:
        pages: Page dicts with keys "title", "url" and "text", as returned by
            `ingest.load_pages`.
        size: Maximum chunk length in characters (`Settings.chunk_size`).
        overlap: Overlap between consecutive chunks (`Settings.chunk_overlap`).

    Returns:
        All chunks of all pages, page by page, in order within each page.
    """
    chunks = []
    for page in pages:
        for i, piece in enumerate(split_fixed(page["text"], size, overlap)):
            chunks.append(Chunk(id=f"{page['title']}#{i}", title=page["title"],
                                url=page["url"], text=piece, position=i))
    return chunks
