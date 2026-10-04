"""PRD Section 3: accumulate_node (Observe).

Sole writer of `evidence_pool`. Appends the current hop's fused candidates without discarding
prior hops' evidence — this is what distinguishes the multi-hop loop from a simple retry: hop 1's
findings about document A are kept while hop 2 goes looking in document B (Section 3).

Runs on every path, including single-hop (where it does one trivial merge into an empty pool,
since there is nothing yet to dedupe against) — kept as its own node rather than folded into
route_retrieve_node so dedup logic has exactly one owner regardless of how many hops ran.
"""

from __future__ import annotations

from app.retrieval.state import RetrievalState


def accumulate_node(state: RetrievalState) -> dict:
    existing_ids = {item.chunk_id for item in state.evidence_pool}
    new_items = [item for item in state.latest_hop_candidates if item.chunk_id not in existing_ids]
    return {"evidence_pool": new_items}
