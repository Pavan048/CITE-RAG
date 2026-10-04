"""PRD Section 3: route_retrieve_node (Act).

Deliberately combines document routing and hybrid chunk retrieval into one node, run fresh on
every hop against the *current* sub-question — never the original question, never a cached
candidate set from a prior hop. This is the specific property that makes multi-document reasoning
work: hop 2 may need to route to an entirely different document than hop 1, so routing cannot be
hoisted out and run once.

Writes only `latest_hop_candidates` (never `evidence_pool` directly) — merging into the pool,
including cross-hop dedup, is accumulate_node's job alone (Section 3), so this node only ever has
to think about retrieval, not accumulation.

Graceful degradation: if the embedding service is unreachable, there is no query vector and
therefore nothing this node can do this hop. On the first hop there is no evidence at all yet, so
this sets `retrieval_unavailable` and short-circuits straight to generation with an honest "service
unavailable" message (see generate_cite.py) — never `not_found`, which would misrepresent an outage
as "the knowledge base doesn't have this." On a later hop, evidence from earlier hops already
exists, so this just contributes nothing for this hop instead of aborting the whole query — the
same treatment as a hop whose routing came up empty.
"""

from __future__ import annotations

import httpx

from app.config import settings
from app.retrieval.state import EvidenceItem, RetrievalState
from app.services.embedding_service import embedding_service
from app.services.qdrant_client import qdrant_service


def route_retrieve_node(state: RetrievalState) -> dict:
    query_text = state.current_sub_question if state.is_multi_hop else state.question
    is_first_hop = state.hop_count == 0

    try:
        query_embedding = embedding_service.embed([query_text])
    except httpx.HTTPError:
        if is_first_hop:
            return {"retrieval_unavailable": True}
        updates: dict = {"latest_hop_candidates": []}
        if state.is_multi_hop:
            updates["hop_count"] = 1
        return updates

    query_dense = query_embedding.dense[0]
    query_sparse = query_embedding.sparse[0]

    updates: dict = {}

    if state.file_ids:
        # Caller pinned the scope — routing is skipped entirely, on every hop (Section 3).
        resolved_doc_ids = list(state.file_ids)
    else:
        doc_results = qdrant_service.search_documents(
            query_dense, top_k=settings.doc_routing_top_k, score_threshold=settings.doc_routing_similarity_floor
        )
        resolved_doc_ids = [r.doc_id for r in doc_results]

        if not resolved_doc_ids and is_first_hop:
            # Nothing cleared the floor on the very first hop: short-circuit to "not found" rather
            # than falling through to an unfiltered, corpus-wide search (Section 3) — a query this
            # far outside the knowledge base on its first attempt is unlikely to be salvaged by
            # burning through the remaining hops on the same unfiltered search.
            return {"not_found": True}

        already_routed = set(state.routed_docs or [])
        newly_routed = [d for d in resolved_doc_ids if d not in already_routed]
        if newly_routed:
            updates["routed_docs"] = newly_routed

        if not resolved_doc_ids:
            # Not the first hop — this specific sub-question's routing just came up empty. Not a
            # global not_found (that short-circuit is "on the first hop" only); this hop simply
            # contributes no evidence, and sufficiency_check_node decides what to do with whatever
            # is already in the pool from other hops. Explicitly cleared (not just omitted) so
            # accumulate_node doesn't re-process a stale list left over from a prior hop.
            updates["latest_hop_candidates"] = []
            if state.is_multi_hop:
                updates["hop_count"] = 1
            return updates

    chunk_results = qdrant_service.hybrid_search_chunks(
        query_dense=query_dense,
        query_sparse=query_sparse,
        doc_ids=resolved_doc_ids,
        limit_per_list=settings.chunk_fusion_top_k,
        rrf_k=settings.rrf_k,
        fused_top_k=settings.chunk_fusion_top_k,
    )

    updates["latest_hop_candidates"] = [
        EvidenceItem(
            chunk_id=r.chunk_id,
            doc_id=r.payload.doc_id,
            page_number=r.payload.page_number,
            section_title=r.payload.section_title,
            content_type=r.payload.content_type,
            parent_block_id=r.payload.parent_block_id,
            text=r.payload.chunk_text,
            fused_score=r.fused_score,
            source_sub_question=query_text,
        )
        for r in chunk_results
    ]
    if state.is_multi_hop:
        updates["hop_count"] = 1
    return updates
