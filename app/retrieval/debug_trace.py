"""Assembles `DebugInfo` (models/schemas.py) from a finished `RetrievalState` plus the graph_path
`run_with_trace` (graph.py) recorded — not part of the PRD's contract, added so the UI's debug
panel can show which nodes ran, what was retrieved at each hop, and every prompt/response, instead
of that only being visible in server logs.
"""

from __future__ import annotations

from app.models.schemas import ChunkTrace, DebugInfo, HopTrace, LLMCallTrace
from app.retrieval.state import EvidenceItem, RetrievalState


def _chunk_trace(item: EvidenceItem, score: float) -> ChunkTrace:
    return ChunkTrace(
        chunk_id=item.chunk_id,
        doc_id=item.doc_id,
        page_number=item.page_number,
        content_type=item.content_type.value,
        score=score,
        text_preview=item.text[:200],
    )


def build_debug_info(state: RetrievalState, graph_path: list[str]) -> DebugInfo:
    # Groups evidence_pool by the sub-question that retrieved it, in the order sub-questions were
    # actually asked — "chunks retrieved per hop", without a separate accumulated state field:
    # evidence_pool already carries `source_sub_question` per item (Section 3). Single-hop queries
    # never populate sub_question_history (only classify_plan_node's multi-hop branch does), so
    # this falls back to treating the original question as hop 1's sub-question in that case.
    by_sub_question: dict[str, list[EvidenceItem]] = {}
    for item in state.evidence_pool:
        by_sub_question.setdefault(item.source_sub_question, []).append(item)

    sub_questions = state.sub_question_history or ([state.question] if state.evidence_pool else [])
    hops = [
        HopTrace(
            hop_number=index + 1,
            sub_question=sub_question,
            chunks_retrieved=[_chunk_trace(item, item.fused_score) for item in by_sub_question.get(sub_question, [])],
        )
        for index, sub_question in enumerate(sub_questions)
    ]

    return DebugInfo(
        graph_path=graph_path,
        is_multi_hop=state.is_multi_hop,
        hop_count=state.hop_count,
        not_found=state.not_found,
        sub_question_history=state.sub_question_history,
        hops=hops,
        reranked_chunks=[_chunk_trace(item, item.rerank_score or 0.0) for item in state.reranked_evidence],
        context_block_count=len(state.context_blocks),
        llm_calls=[LLMCallTrace(node=c.node, prompt=c.prompt, response=c.response) for c in state.llm_calls],
    )
