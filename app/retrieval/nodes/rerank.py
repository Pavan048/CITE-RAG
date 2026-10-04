"""PRD Section 3: rerank_node. Operates over the *entire* evidence_pool (potentially spanning
multiple hops and documents), not a single hop's results — reranking runs once, after the loop
ends, against the pool as a whole.

Scores against the *original* question, same reasoning as sufficiency_check_node: the pool may
hold evidence gathered for several different sub-questions, but final relevance is judged against
what the user actually asked.

Produces `reranked_evidence` (sorted best-first, trimmed to `top_k`) rather than mutating
`evidence_pool` in place — see the field's docstring in retrieval/state.py for why.

Graceful degradation: reranking is a quality improvement over the RRF-fused order (Section 3),
not a correctness requirement — every candidate already has a `fused_score` from route_retrieve_node.
If the reranker service is unreachable, this falls back to sorting by that fused order rather than
failing the whole query — a worse ranking is a better outcome than an outage over what is, in the
end, a refinement step.
"""

from __future__ import annotations

import httpx

from app.retrieval.state import RetrievalState
from app.services.reranker_service import reranker_service


def rerank_node(state: RetrievalState) -> dict:
    if not state.evidence_pool:
        return {"reranked_evidence": []}

    texts = [item.text for item in state.evidence_pool]
    try:
        results = reranker_service.rerank(state.question, texts)  # sorted best-first
    except httpx.HTTPError:
        fallback = sorted(state.evidence_pool, key=lambda item: item.fused_score, reverse=True)
        return {"reranked_evidence": fallback[: state.top_k]}

    scored = [state.evidence_pool[r.index].model_copy(update={"rerank_score": r.score}) for r in results]
    return {"reranked_evidence": scored[: state.top_k]}
