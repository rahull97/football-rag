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
    """Claude's answer plus everything needed to show and audit it.

    Attributes:
        text: Claude's answer, with citation markers like [1] that refer to
            positions in `sources` (1-based). Equals `NO_ANSWER` when Claude
            follows the "I don't know" rule exactly.
        sources: The chunks given to Claude, in the order they were numbered.
            `sources[0]` is [1].
        input_tokens: Prompt tokens billed for the call (system prompt +
            sources + question).
        output_tokens: Tokens Claude generated.
    """

    text: str
    sources: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as the numbered source list Claude sees.

    Each chunk becomes a block like:

        [1] (from "Leicester City F.C.")
        <chunk text>

    Blocks are separated by a blank line and numbered from 1 in list order.
    The numbers are what Claude cites, and they match the numbering the CLI
    prints under "Sources:".

    Args:
        chunks: Retrieved chunks, most relevant first.

    Returns:
        The formatted text (empty string if `chunks` is empty).
    """
    return "\n\n".join(f"[{i}] (from \"{c.title}\")\n{c.text}" for i, c in enumerate(chunks, 1))


def generate_answer(client: Anthropic, model: str, question: str,
                    chunks: list[RetrievedChunk]) -> Answer:
    """Ask Claude to answer `question` using only `chunks`, with citations.

    Makes one Messages API call. The system prompt (`SYSTEM_PROMPT`) sets the
    grounding rules; the user message contains the numbered sources inside
    <sources>...</sources> tags followed by the question. The answer is capped
    at 500 output tokens. All text blocks of the reply are joined and stripped.

    This function does not check the answer against the sources; citation
    accuracy is only measured indirectly by the LLM judge in `evaluate.py`.

    Args:
        client: An Anthropic client (reads ANTHROPIC_API_KEY).
        model: Claude model id (`Settings.answer_model`).
        question: The user's question.
        chunks: Retrieved chunks to use as sources.

    Returns:
        An `Answer` with the text, the chunks used, and token usage.

    Raises:
        anthropic.APIError: Any API failure (missing/invalid key,
            rate limit, unknown model, network error) is passed through.
    """
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
