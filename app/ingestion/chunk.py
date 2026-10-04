"""PRD Section 2 Stage 3: chunk + doc summary.

Chunking is recursive and structure-aware (Section 1): each document section (the contiguous run
of elements between one heading and the next) becomes one parent block, and within a section, text
elements are greedily windowed to ~512 tokens with 10-15% overlap, splitting at paragraph
boundaries first and only falling back to sentence- or hard-token-level splitting for a single
paragraph that alone exceeds the target (`_split_oversized_unit` below) — never truncating mid-word
or discarding structure when a paragraph boundary would do. Tables and figure captions are already
atomic, self-contained units (Section 2 Stage 2) and become their own single chunk each, never
merged into the surrounding text windows.

The prompt-injection mitigation (Section 8) runs here, at "the point Docling output is converted to
chunks" as specified, on every element's text before it becomes part of either a chunk or a parent
block — both are eventually placed into a generation prompt (context_budget_node swaps a chunk for
its parent block), so both must be sanitized, not just the child chunk text.
"""

from __future__ import annotations

import hashlib
import re
import uuid

import tiktoken

from app.config import settings
from app.extensibility.prompt_provider import prompt_provider
from app.ingestion.parse import ParsedDocument, ParsedElement
from app.models.mongo_models import ChunkPayload, ContentType, ParentBlockRecord
from app.services.llm_client import llm_client

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Narrow, cheap mitigation (Section 8) — not a full guardrail layer, just enough to keep the most
# obvious instruction-hijacking phrasing out of text that will later sit inside a generation
# prompt. Matches are redacted, not the whole chunk dropped, so legitimate surrounding content
# survives.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the) (previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all|any|the) (previous|prior|above) (instructions|context)", re.IGNORECASE),
    re.compile(r"you are now (a|an|in) .{0,40}(mode|persona)", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"new instructions?\s*:", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(settings.token_count_encoding)


def _greedy_windows(
    units: list[tuple[str, int]], target_tokens: int, overlap_ratio: float, encoding: tiktoken.Encoding
) -> list[tuple[str, int]]:
    """Merge (text, page_number) units into ~target_tokens windows with trailing-unit overlap
    carried into the next window. Each returned window is tagged with the page_number of whichever
    unit it *starts* with — a window that straddles a page break is cited to where it begins."""
    windows: list[tuple[str, int]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0

    for unit_text, page_number in units:
        unit_tokens = len(encoding.encode(unit_text))
        if current and current_tokens + unit_tokens > target_tokens:
            windows.append((" ".join(t for t, _ in current), current[0][1]))
            overlap_budget = int(target_tokens * overlap_ratio)
            carried: list[tuple[str, int]] = []
            carried_tokens = 0
            for u_text, u_page in reversed(current):
                t = len(encoding.encode(u_text))
                if carried_tokens + t > overlap_budget:
                    break
                carried.insert(0, (u_text, u_page))
                carried_tokens += t
            current, current_tokens = carried, carried_tokens
        current.append((unit_text, page_number))
        current_tokens += unit_tokens

    if current:
        windows.append((" ".join(t for t, _ in current), current[0][1]))
    return windows


def _split_oversized_unit(unit: str, page_number: int, target_tokens: int, encoding: tiktoken.Encoding) -> list[tuple[str, int]]:
    """Recursion fallback for a single paragraph that alone exceeds the token target: try
    sentence-level splitting first, and only hard-slice by raw tokens if a single sentence is
    itself still too long."""
    if len(encoding.encode(unit)) <= target_tokens:
        return [(unit, page_number)]
    sentences = _SENTENCE_SPLIT_RE.split(unit)
    if len(sentences) > 1:
        return _greedy_windows([(s, page_number) for s in sentences], target_tokens, overlap_ratio=0.0, encoding=encoding)
    tokens = encoding.encode(unit)
    return [(encoding.decode(tokens[i : i + target_tokens]), page_number) for i in range(0, len(tokens), target_tokens)]


def _group_into_sections(elements: list[ParsedElement]) -> list[list[ParsedElement]]:
    sections: list[list[ParsedElement]] = []
    current: list[ParsedElement] = []
    current_title: str | None | object = object()  # sentinel distinct from any real title (incl. None)
    for el in elements:
        if el.section_title != current_title:
            if current:
                sections.append(current)
            current = []
            current_title = el.section_title
        current.append(el)
    if current:
        sections.append(current)
    return sections


def chunk_document(parsed: ParsedDocument, doc_id: str) -> tuple[list[tuple[str, ChunkPayload]], list[ParentBlockRecord]]:
    encoding = _encoding()
    chunks: list[tuple[str, ChunkPayload]] = []
    parent_blocks: list[ParentBlockRecord] = []
    chunk_index = 0

    for section_elements in _group_into_sections(parsed.elements):
        sanitized = [(el, sanitize_text(el.text)) for el in section_elements]
        parent_block_id = str(uuid.uuid4())
        section_title = section_elements[0].section_title
        parent_blocks.append(
            ParentBlockRecord(parent_block_id=parent_block_id, doc_id=doc_id, text="\n\n".join(t for _, t in sanitized))
        )

        # Tables and image captions are already atomic (Section 2 Stage 2) — one chunk each,
        # never merged into the text-window pass below.
        text_units: list[tuple[ParsedElement, str]] = []
        for el, text in sanitized:
            if el.content_type != ContentType.TEXT:
                chunks.append(
                    _make_chunk(doc_id, chunk_index, el.page_number, section_title, el.content_type, parent_block_id, text, encoding)
                )
                chunk_index += 1
            else:
                text_units.append((el, text))

        if not text_units:
            continue

        expanded_units: list[tuple[str, int]] = []
        for el, text in text_units:
            expanded_units.extend(_split_oversized_unit(text, el.page_number, settings.chunk_target_tokens, encoding))

        windows = _greedy_windows(expanded_units, settings.chunk_target_tokens, settings.chunk_overlap_ratio, encoding)
        for window_text, page_number in windows:
            chunks.append(
                _make_chunk(doc_id, chunk_index, page_number, section_title, ContentType.TEXT, parent_block_id, window_text, encoding)
            )
            chunk_index += 1

    return chunks, parent_blocks


def _make_chunk(
    doc_id: str, chunk_index: int, page_number: int, section_title: str | None,
    content_type: ContentType, parent_block_id: str, text: str, encoding: tiktoken.Encoding,
) -> tuple[str, ChunkPayload]:
    # Deterministic, not random — re-running the pipeline for the same doc_id (e.g. retrying after
    # a partial indexing failure, Section 2 Stage 5) upserts the same points instead of creating
    # duplicates. Random uuid4 ids would make every retry a fresh set of orphaned points.
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))
    payload = ChunkPayload(
        doc_id=doc_id,
        page_number=page_number,
        section_title=section_title,
        content_type=content_type,
        chunk_index=chunk_index,
        parent_block_id=parent_block_id,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        token_count=len(encoding.encode(text)),
        chunk_text=text,
    )
    return chunk_id, payload


def generate_document_summary(parsed: ParsedDocument, filename: str, llm_api_key: str, llm_model: str) -> str:
    source_text = parsed.full_markdown[: settings.doc_summary_source_max_chars]
    prompt = prompt_provider.get(
        "ingestion.document_summary",
        filename=filename,
        document_text=source_text,
        min_tokens=settings.doc_summary_min_tokens,
        max_tokens=settings.doc_summary_max_tokens,
    )
    # `llm_api_key` used for exactly this one call, never retained beyond this function (Section 8).
    return llm_client.complete_text(prompt=prompt, api_key=llm_api_key, model=llm_model)
