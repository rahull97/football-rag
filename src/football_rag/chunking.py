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
    id: str          # unique id, e.g. "Liverpool F.C.#12"
    title: str       # article the chunk came from
    url: str
    text: str
    position: int    # index of the chunk within its article


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into windows of about `size` characters.

    Consecutive windows share `overlap` characters, so a fact that sits on a
    boundary still appears whole in at least one chunk. Cuts are moved back to
    the nearest whitespace so words are not split in half.
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
    chunks = []
    for page in pages:
        for i, piece in enumerate(split_fixed(page["text"], size, overlap)):
            chunks.append(Chunk(id=f"{page['title']}#{i}", title=page["title"],
                                url=page["url"], text=piece, position=i))
    return chunks
