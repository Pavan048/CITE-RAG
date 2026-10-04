"""PRD Section 3: generate_cite_node.

Loads its prompt via `PromptProvider.get("generation.grounded_answer")`. The model is asked to
write inline `[doc_id:page_number:chunk_id]` markers, but only the `chunk_id` inside each marker is
ever trusted — after generation, every marker is checked against the known-good
`(chunk_id -> doc_id, page_number)` mapping built from `context_blocks` (i.e. exactly the context
that was actually supplied, post-rerank/budget), and `doc_id`/`page_number` in the final output are
always re-derived from that trusted mapping, never taken from what the model typed. A marker whose
chunk_id isn't in that mapping is stripped from the answer text entirely and never becomes a
structured citation — dropped, not "trusted anyway" (Section 3).

If there is no context at all (context_blocks is empty — whether because route_retrieve_node's
short-circuit set `not_found`, or the loop simply never found anything to rerank), this skips the
LLM call entirely and returns a fixed abstention answer. There is nothing to hallucinate from
empty context, but there is also no reason to spend a call asking a model to confirm what is
already known with certainty.

`retrieval_unavailable` (the embedding service was unreachable on the first hop) gets its own,
different fixed message rather than reusing `NOT_FOUND_ANSWER` — "nothing relevant in the
knowledge base" and "the retrieval service is down" are different failure modes, and telling the
user the former when the truth is the latter would be actively misleading, not just imprecise.

`build_grounded_answer_prompt` and `resolve_citations` are exported (not module-private) because
api/query.py's streaming endpoint needs the exact same prompt-building and citation-validation
logic for its streamed generation call — the only thing that differs between the blocking node
below and the streaming path is whether `llm_client.complete_text` or `llm_client.stream_text` is
what actually talks to the model. Duplicating this logic for the streaming path would risk the two
citation-validation implementations drifting apart, which is exactly the kind of thing that's hard
to catch in review (PRD's own words for this general class of risk).
"""

from __future__ import annotations

import re

from app.config import settings
from app.extensibility.prompt_provider import prompt_provider
from app.models.schemas import Citation
from app.retrieval.state import ContextBlock, LLMCallRecord, RetrievalState
from app.services.llm_client import llm_client

_CITATION_MARKER_RE = re.compile(r"\[([^\[\]:]+):(\d+):([^\[\]:]+)\]")

NOT_FOUND_ANSWER = "I could not find relevant information in the knowledge base to answer this question."
RETRIEVAL_UNAVAILABLE_ANSWER = "The retrieval service is temporarily unavailable. Please try again in a moment."


def build_grounded_answer_prompt(state: RetrievalState) -> str:
    return prompt_provider.get("generation.grounded_answer", question=state.question, context_blocks=state.context_blocks)


def resolve_citations(raw_answer: str, context_blocks: list[ContextBlock]) -> tuple[str, list[Citation]]:
    """Validates every inline citation marker in `raw_answer` against `context_blocks` (the exact
    context actually supplied, post-rerank/budget) and rewrites `doc_id`/`page_number` from that
    trusted data. See module docstring for why this can never trust what the model typed."""
    trusted: dict[str, tuple[str, int, str]] = {}
    for block in context_blocks:
        for option in block.citation_options:
            trusted[option.chunk_id] = (block.doc_id, option.page_number, block.text)

    citations: list[Citation] = []
    seen_chunk_ids: set[str] = set()

    def _resolve(match: re.Match) -> str:
        chunk_id = match.group(3)
        if chunk_id not in trusted:
            return ""  # drop: points to a chunk_id that was never actually supplied
        doc_id, page_number, block_text = trusted[chunk_id]
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            citations.append(
                Citation(doc_id=doc_id, page_number=page_number, chunk_id=chunk_id, snippet=block_text[: settings.citation_snippet_max_chars])
            )
        return f"[{doc_id}:{page_number}:{chunk_id}]"

    cleaned_answer = _CITATION_MARKER_RE.sub(_resolve, raw_answer)
    return cleaned_answer, citations


def generate_cite_node(state: RetrievalState) -> dict:
    if state.retrieval_unavailable:
        return {"answer": RETRIEVAL_UNAVAILABLE_ANSWER, "citations": []}
    if not state.context_blocks:
        return {"answer": NOT_FOUND_ANSWER, "citations": []}

    prompt = build_grounded_answer_prompt(state)
    raw_answer = llm_client.complete_text(prompt=prompt, api_key=state.llm_api_key, model=state.llm_model)
    cleaned_answer, citations = resolve_citations(raw_answer, state.context_blocks)

    return {
        "answer": cleaned_answer,
        "citations": citations,
        "llm_calls": [LLMCallRecord(node="generate_cite", prompt=prompt, response=raw_answer)],
    }
