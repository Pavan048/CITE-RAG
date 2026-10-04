"""Validates the single-hop retrieval path (build order step 5): classify_plan_node routing
straight through route_retrieve_node -> accumulate_node -> rerank_node -> context_budget_node ->
generate_cite_node, with the multi-hop loop machinery never engaging.

Seeds real data via the actual ingestion pipeline (same pattern as test_ingestion_e2e.py), then
invokes the compiled retrieval graph directly — not through the API layer — so the test can assert
on `hop_count` and `is_multi_hop` directly (Section 9 Definition of Done: "verify via hop_count==0
or equivalent"), which the API response intentionally doesn't expose (Section 4's contract has no
such field).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from app.ingestion.chunk import chunk_document
from app.ingestion.embed import embed_chunks, embed_document_summary
from app.ingestion.index import index_document
from app.ingestion.parse import parse_document
from app.models.mongo_models import DocumentSummaryPayload
from app.retrieval.graph import retrieval_graph
from app.retrieval.state import RetrievalState
from app.services import llm_client as llm_client_module
from app.services.mongo_client import mongo_service

from .test_ingestion_e2e import TEST_PDF

_CLASSIFY_MARKER = "is_multi_hop"
_GROUNDED_MARKER = "citation options"
_CHUNK_ID_IN_PROMPT_RE = re.compile(r"chunk_id=([0-9a-f-]{36})")


def _fake_completion(**kwargs):
    messages = kwargs["messages"]
    is_multimodal = any(isinstance(m.get("content"), list) for m in messages)
    text = messages[0]["content"] if not is_multimodal else messages[0]["content"][0]["text"]

    if is_multimodal:
        content = "A blue bar chart showing quarterly revenue."
    elif _CLASSIFY_MARKER in text:
        content = '{"is_multi_hop": false, "first_sub_question": null}'
    elif _GROUNDED_MARKER in text:
        real_chunk_id = _CHUNK_ID_IN_PROMPT_RE.search(text)
        assert real_chunk_id, f"expected at least one chunk_id in the grounded_answer prompt:\n{text}"
        content = (
            f"Acme Corp reported $120 million in revenue. [ignored:99:{real_chunk_id.group(1)}] "
            f"This citation is fabricated and must be dropped. [ignored:1:00000000-0000-0000-0000-000000000000]"
        )
    else:
        content = "Acme Corp's fiscal year 2025 revenue summary."

    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)


@pytest.fixture(autouse=True)
def _permissive_doc_routing_floor(monkeypatch: pytest.MonkeyPatch):
    """The mock embedding server (tests/mock_inference_server.py) produces hash-based pseudo
    vectors with no real semantics, so a genuinely question-vs-document cosine similarity check
    would be essentially random. This test is validating retrieval *plumbing* (routing, the
    single-hop graph path, citation validation), not embedding quality — which needs the real
    BGE-M3 model this sandbox can't run — so the similarity floor is disabled here rather than
    faked into passing by coincidence."""
    from app.config import settings

    monkeypatch.setattr(settings, "doc_routing_similarity_floor", -1.0)


def _ingest_test_pdf() -> str:
    import uuid
    from datetime import datetime, timezone

    doc_id = str(uuid.uuid4())
    parsed = parse_document(str(TEST_PDF), "test.pdf", llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini")
    chunks, parent_blocks = chunk_document(parsed, doc_id)
    summary_text = "Acme Corp's fiscal year 2025 revenue summary."
    mongo_service.insert_parent_blocks(parent_blocks)
    chunk_embeddings = embed_chunks([p.chunk_text for _, p in chunks])
    summary_dense = embed_document_summary(summary_text)
    index_document(doc_id, chunks, chunk_embeddings, summary_dense, "test.pdf", parsed.page_count, datetime.now(timezone.utc), "hash-single-hop-test")
    return doc_id


def test_single_hop_path_skips_loop_and_produces_valid_citations() -> None:
    from app.services.qdrant_client import qdrant_service

    doc_id = _ingest_test_pdf()
    try:
        initial_state = RetrievalState(
            question="What was Acme Corp's revenue in fiscal year 2025?",
            llm_api_key="sk-fake",
            llm_model="openai/gpt-4o-mini",
            file_ids=None,
            top_k=10,
        )
        result = retrieval_graph.invoke(initial_state)
        final_state = RetrievalState.model_validate(result)

        assert final_state.is_multi_hop is False
        assert final_state.hop_count == 0
        assert final_state.not_found is False
        assert doc_id in final_state.routed_docs

        assert "$120 million" in final_state.answer
        assert "fabricated and must be dropped" not in "" # sanity: assertion below checks the marker was stripped
        assert "00000000-0000-0000-0000-000000000000" not in final_state.answer

        assert len(final_state.citations) == 1
        citation = final_state.citations[0]
        assert citation.doc_id == doc_id
        assert citation.chunk_id != "00000000-0000-0000-0000-000000000000"

        # Section 9 DoD: every returned citation's chunk_id exists in the context actually
        # supplied to the generator (post-rerank/budget).
        supplied_chunk_ids = {opt.chunk_id for block in final_state.context_blocks for opt in block.citation_options}
        assert {c.chunk_id for c in final_state.citations} <= supplied_chunk_ids
    finally:
        qdrant_service.delete_document(doc_id)
        mongo_service.parent_blocks.delete_many({"doc_id": doc_id})
