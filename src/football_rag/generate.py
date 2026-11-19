"""Step 5 of RAG: ask Claude to answer using only the retrieved chunks.

The prompt numbers each chunk as a source, tells Claude to cite them like
[1], and to say it doesn't know when the sources don't contain the answer.
That last instruction is what stops the bot from making things up.
"""

from dataclasses import dataclass

from anthropic import Anthropic

from .store import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about football using ONLY the numbered sources provided.

Rules:
- Base every statement on the sources. Do not use outside knowledge.
- Cite the sources you used in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I don't know based on the available sources."
- Be concise: one to three sentences unless the question needs more."""

NO_ANSWER = "I don't know based on the available sources."


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] (from \"{c.title}\")\n{c.text}" for i, c in enumerate(chunks, 1))


def generate_answer(client: Anthropic, model: str, question: str,
                    chunks: list[RetrievedChunk]) -> Answer:
    user_message = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}"
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    return Answer(text=text, sources=chunks, input_tokens=resp.usage.input_tokens,
                  output_tokens=resp.usage.output_tokens)
