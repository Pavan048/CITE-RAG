"""PRD Section 3: context_budget_node.

Swaps each surviving chunk for its parent block (Section 2: hierarchical chunking stores a full
parent block alongside every child chunk specifically so generation gets fuller context than the
small embedding-sized chunk). Multiple surviving chunks that share a parent are deduped down to
one context block — otherwise the same parent text could appear two or three times purely because
several of its child chunks scored well, wasting token budget on literal duplication.

That dedup creates a real question, though: citations must name a specific chunk_id + page_number
(Section 3), but a deduped block's *text* is the parent's, which can span multiple original chunks
and page numbers. The resolution here (confirmed with the user) is that each block carries a list
of `citation_options` — every (chunk_id, page_number) pair that legitimately backs that block's
text — and generate_cite_node picks whichever option best fits each sentence. Citation validation
downstream only needs to check a cited chunk_id against this known-good set; it never has to trust
page_number or doc_id text the model typed inline (those get re-derived from this same trusted
data), so an imprecise-but-real citation is never a fabricated one.

Fits the surviving blocks under a hard token cap, dropping lowest-reranked blocks first when
trimming and never truncating a block's text (Section 3) — trimming operates at whole-block
granularity only.
"""

from __future__ import annotations

import tiktoken

from app.config import settings
from app.retrieval.state import CitationOption, ContextBlock, EvidenceItem, RetrievalState
from app.services.mongo_client import mongo_service


def context_budget_node(state: RetrievalState) -> dict:
    if not state.reranked_evidence:
        return {"context_blocks": []}

    parent_block_ids = list({item.parent_block_id for item in state.reranked_evidence})
    parent_texts = mongo_service.get_parent_blocks(parent_block_ids)

    groups: dict[str, list[EvidenceItem]] = {}
    for item in state.reranked_evidence:
        groups.setdefault(item.parent_block_id, []).append(item)

    encoding = tiktoken.get_encoding(settings.token_count_encoding)
    candidates: list[tuple[float, int, ContextBlock]] = []
    for parent_block_id, items in groups.items():
        # Falls back to the chunk's own text if a parent block somehow isn't found (should not
        # happen in practice — every chunk is written with a parent_block_id at ingestion time).
        text = parent_texts.get(parent_block_id) or items[0].text
        best_score = max((i.rerank_score or 0.0) for i in items)
        block = ContextBlock(
            doc_id=items[0].doc_id,
            text=text,
            citation_options=[CitationOption(chunk_id=i.chunk_id, page_number=i.page_number) for i in items],
        )
        candidates.append((best_score, len(encoding.encode(text)), block))

    candidates.sort(key=lambda c: c[0], reverse=True)

    kept: list[ContextBlock] = []
    total_tokens = 0
    for _score, token_count, block in candidates:
        if kept and total_tokens + token_count > settings.context_token_budget:
            # Everything from here on is lower-ranked than what's already kept (sorted desc) —
            # stop rather than let a smaller-but-lower-ranked block jump the queue.
            break
        total_tokens += token_count
        kept.append(block)

    return {"context_blocks": kept}
