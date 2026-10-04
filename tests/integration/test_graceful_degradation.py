"""Validates the graceful-degradation behavior added after a production-readiness review:
  - rerank_node falls back to the RRF-fused order instead of failing the query when the reranker
    service is unreachable (app/retrieval/nodes/rerank.py).
  - route_retrieve_node treats an unreachable embedding service as `retrieval_unavailable` (first
    hop, no evidence exists yet to fall back on) or as an empty hop (later hop, evidence from
    earlier hops still exists) — never as `not_found`, which would misrepresent an outage as "the
    knowledge base doesn't have this" (app/retrieval/nodes/route_retrieve.py,
    app/retrieval/nodes/generate_cite.py).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.ingestion.chunk import chunk_document
from app.ingestion.embed import embed_chunks, embed_document_summary
from app.ingestion.index import index_document
from app.ingestion.parse import parse_document
from app.retrieval.graph import retrieval_graph
from app.retrieval.nodes.generate_cite import RETRIEVAL_UNAVAILABLE_ANSWER
from app.retrieval.state import RetrievalState
from app.services import embedding_service as embedding_service_module
from app.services import llm_client as llm_client_module
from app.services import reranker_service as reranker_service_module
from app.services.mongo_client import mongo_service
from app.services.qdrant_client import qdrant_service

from .test_ingestion_e2e import TEST_PDF

TEST_PDF_BETA = TEST_PDF.parent / "test_beta.pdf"
_CHUNK_ID_IN_PROMPT_RE = re.compile(r"chunk_id=([0-9a-f-]{36})")


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_completion(**kwargs):
    messages = kwargs["messages"]
    is_multimodal = any(isinstance(m.get("content"), list) for m in messages)
    text = messages[0]["content"] if not is_multimodal else messages[0]["content"][0]["text"]

    if is_multimodal:
        return _llm_response("A chart.")
    if "is_multi_hop" in text:
        return _llm_response('{"is_multi_hop": false, "first_sub_question": null}')
    if "citation options" in text:
        match = _CHUNK_ID_IN_PROMPT_RE.search(text)
        assert match, f"expected a chunk_id in the grounded_answer prompt:\n{text}"
        return _llm_response(f"Acme Corp reported $120 million in revenue. [ignored:1:{match.group(1)}]")
    return _llm_response("summary")


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)


@pytest.fixture(autouse=True)
def _permissive_doc_routing_floor(monkeypatch: pytest.MonkeyPatch):
    from app.config import settings

    monkeypatch.setattr(settings, "doc_routing_similarity_floor", -1.0)


def _ingest(pdf_path, filename: str, summary_text: str, content_hash: str) -> str:
    doc_id = str(uuid.uuid4())
    parsed = parse_document(str(pdf_path), filename, llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini")
    chunks, parent_blocks = chunk_document(parsed, doc_id)
    mongo_service.insert_parent_blocks(parent_blocks)
    chunk_embeddings = embed_chunks([p.chunk_text for _, p in chunks])
    summary_dense = embed_document_summary(summary_text)
    index_document(doc_id, chunks, chunk_embeddings, summary_dense, filename, parsed.page_count, datetime.now(timezone.utc), content_hash)
    return doc_id


def _cleanup(*doc_ids: str) -> None:
    for doc_id in doc_ids:
        qdrant_service.delete_document(doc_id)
        mongo_service.parent_blocks.delete_many({"doc_id": doc_id})


def test_rerank_falls_back_to_fused_order_when_reranker_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = _ingest(TEST_PDF, "acme.pdf", "Acme Corp fiscal year 2025 revenue report.", "hash-degradation-rerank")
    try:
        def _broken_rerank(*args, **kwargs):
            raise httpx.ConnectError("reranker is down")

        monkeypatch.setattr(reranker_service_module.reranker_service, "rerank", _broken_rerank)

        initial_state = RetrievalState(
            question="What was Acme Corp's revenue in fiscal year 2025?",
            llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini", file_ids=None, top_k=10,
        )
        result = retrieval_graph.invoke(initial_state)
        final_state = RetrievalState.model_validate(result)

        # The query still succeeds end to end — a worse ranking, not an outage.
        assert "$120 million" in final_state.answer
        assert len(final_state.citations) == 1
        assert final_state.reranked_evidence, "expected the fused-order fallback to populate reranked_evidence"
        # Falls back to fused_score order: descending, and rerank_score was never set.
        scores = [item.fused_score for item in final_state.reranked_evidence]
        assert scores == sorted(scores, reverse=True)
        assert all(item.rerank_score is None for item in final_state.reranked_evidence)
    finally:
        _cleanup(doc_id)


def test_embedding_outage_on_first_hop_is_reported_distinctly_from_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = _ingest(TEST_PDF, "acme.pdf", "Acme Corp fiscal year 2025 revenue report.", "hash-degradation-embed-hop1")
    try:
        def _broken_embed(*args, **kwargs):
            raise httpx.ConnectError("embedding service is down")

        monkeypatch.setattr(embedding_service_module.embedding_service, "embed", _broken_embed)

        initial_state = RetrievalState(
            question="What was Acme Corp's revenue in fiscal year 2025?",
            llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini", file_ids=None, top_k=10,
        )
        result = retrieval_graph.invoke(initial_state)
        final_state = RetrievalState.model_validate(result)

        assert final_state.retrieval_unavailable is True
        assert final_state.not_found is False  # must not be conflated with "nothing relevant"
        assert final_state.answer == RETRIEVAL_UNAVAILABLE_ANSWER
        assert final_state.citations == []
    finally:
        _cleanup(doc_id)


def test_embedding_outage_on_a_later_hop_keeps_earlier_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    acme_id = _ingest(TEST_PDF, "acme.pdf", "Acme Corp fiscal year 2025 revenue report.", "hash-degradation-embed-hop2-a")
    beta_id = _ingest(TEST_PDF_BETA, "beta.pdf", "Beta Inc fiscal year 2025 revenue report.", "hash-degradation-embed-hop2-b")
    try:
        def _multi_hop_completion(**kwargs):
            text = kwargs["messages"][0]["content"]
            if "is_multi_hop" in text:
                return _llm_response('{"is_multi_hop": true, "first_sub_question": "What was Acme revenue?"}')
            if "Accumulated evidence" in text:
                # Only Acme's evidence will ever be gathered (hop 2's embedding call fails before
                # Beta is ever routed to) — sufficiency must eventually give up via the hop cap,
                # not hang, and it must do so on whatever hop 1 alone provided.
                return _llm_response('{"sufficient": false, "missing": "still need Beta Inc revenue"}')
            if "next_sub_question" in text:
                return _llm_response('{"next_sub_question": "What was Beta Inc revenue?"}')
            if "citation options" in text:
                match = _CHUNK_ID_IN_PROMPT_RE.search(text)
                assert match
                return _llm_response(f"Acme Corp reported $120 million. [ignored:1:{match.group(1)}]")
            return _llm_response("summary")

        monkeypatch.setattr(llm_client_module.litellm, "completion", _multi_hop_completion)

        real_embed = embedding_service_module.embedding_service.embed
        call_count = {"n": 0}

        def _fails_from_second_call(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise httpx.ConnectError("embedding service is down")
            return real_embed(*args, **kwargs)

        monkeypatch.setattr(embedding_service_module.embedding_service, "embed", _fails_from_second_call)

        initial_state = RetrievalState(
            question="Compare Acme Corp's and Beta Inc's revenue.",
            llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini", file_ids=None, top_k=10,
        )
        result = retrieval_graph.invoke(initial_state)
        final_state = RetrievalState.model_validate(result)

        # Hop 1 succeeded before the outage; the query still reaches a (partial) answer rather
        # than aborting outright the moment hop 2's embedding call fails.
        assert final_state.retrieval_unavailable is False
        assert any(item.doc_id == acme_id for item in final_state.evidence_pool)
        assert final_state.partial_answer is True  # hop cap reached without ever finding Beta's evidence
        assert "$120 million" in final_state.answer
    finally:
        _cleanup(acme_id, beta_id)
