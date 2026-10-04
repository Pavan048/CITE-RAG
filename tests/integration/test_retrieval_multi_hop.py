"""Validates the multi-hop loop (build order step 6): accumulate_node/sufficiency_check_node, the
loop-back edge, and the hop cap — on top of the already-working single-hop path.

Two Definition of Done bullets (Section 9) drive the two tests here:
  - "Multi-document comparison query against two unrelated documents -> sub_question_history shows
    at least two distinct sub-questions, routed_docs shows candidates from both documents across
    hops, not just one."
  - "Force a query designed to never reach sufficiency -> response has partial_answer: true at hop
    4, not a confidently wrong complete-looking answer."
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.config import settings
from app.ingestion.chunk import chunk_document
from app.ingestion.embed import embed_chunks, embed_document_summary
from app.ingestion.index import index_document
from app.ingestion.parse import parse_document
from app.retrieval.graph import retrieval_graph
from app.retrieval.state import RetrievalState
from app.services import llm_client as llm_client_module
from app.services.mongo_client import mongo_service
from app.services.qdrant_client import qdrant_service

from .test_ingestion_e2e import TEST_PDF

TEST_PDF_BETA = TEST_PDF.parent / "test_beta.pdf"

_ACME_SUBQ = "What was Acme Corp's revenue growth in fiscal year 2025?"
_BETA_SUBQ = "What was Beta Inc's revenue growth in fiscal year 2025?"
_CHUNK_ID_RE = re.compile(r"chunk_id=([0-9a-f-]{36})")


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _ingest(pdf_path, filename: str, summary_text: str, content_hash: str) -> str:
    doc_id = str(uuid.uuid4())
    parsed = parse_document(str(pdf_path), filename, llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini")
    chunks, parent_blocks = chunk_document(parsed, doc_id)
    mongo_service.insert_parent_blocks(parent_blocks)
    chunk_embeddings = embed_chunks([p.chunk_text for _, p in chunks])
    summary_dense = embed_document_summary(summary_text)
    index_document(doc_id, chunks, chunk_embeddings, summary_dense, filename, parsed.page_count, datetime.now(timezone.utc), content_hash)
    return doc_id


def _ingestion_time_completion(**kwargs):
    """Covers the LLM calls ingestion itself makes (figure captioning, doc summary) — separate
    from each test's own mock of the *retrieval*-time calls, and applied first (see
    `two_documents` below) since fixture setup runs before a test's own monkeypatch.setattr."""
    is_multimodal = any(isinstance(m.get("content"), list) for m in kwargs["messages"])
    content = "A chart image." if is_multimodal else "fiscal year 2025 revenue and growth report."
    return _llm_response(content)


@pytest.fixture()
def two_documents(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _ingestion_time_completion)
    acme_id = _ingest(TEST_PDF, "acme.pdf", "Acme Corp fiscal year 2025 revenue and growth report.", "hash-acme-multihop")
    beta_id = _ingest(TEST_PDF_BETA, "beta.pdf", "Beta Inc fiscal year 2025 revenue and growth report.", "hash-beta-multihop")
    yield acme_id, beta_id
    for doc_id in (acme_id, beta_id):
        qdrant_service.delete_document(doc_id)
        mongo_service.parent_blocks.delete_many({"doc_id": doc_id})


@pytest.fixture(autouse=True)
def _permissive_doc_routing(monkeypatch: pytest.MonkeyPatch):
    # Same reasoning as test_retrieval_single_hop.py: the mock embedding server has no learned
    # semantics, only word-overlap-driven similarity, so the floor is loosened rather than relied
    # on to pass by coincidence — what's under test here is the loop, not embedding quality.
    #
    # doc_routing_top_k is also pinned to 1 for these two tests specifically: with a corpus of
    # only two documents, the *default* top_k (30) would return both documents on every hop
    # regardless of which sub-question is asked, making it impossible to observe "hop 1 routes to
    # document A, hop 2 routes to document B" — the exact distinguishing behavior under test.
    # Forcing top-1 makes each hop's routing decision meaningful even with the mock embedding's
    # crude bag-of-words similarity (Acme's sub-question still scores Acme's summary above Beta's,
    # since "Acme"/"Corp" are shared with only one of the two summaries).
    monkeypatch.setattr(settings, "doc_routing_similarity_floor", -1.0)
    monkeypatch.setattr(settings, "doc_routing_top_k", 1)


def _is_classify_plan_prompt(text: str) -> bool:
    return "is_multi_hop" in text


def _is_sufficiency_prompt(text: str) -> bool:
    return "Accumulated evidence" in text


def _is_rewrite_prompt(text: str) -> bool:
    return "next_sub_question" in text


def _is_grounded_answer_prompt(text: str) -> bool:
    return "citation options" in text


def test_multi_hop_comparison_routes_to_both_documents(monkeypatch: pytest.MonkeyPatch, two_documents) -> None:
    acme_id, beta_id = two_documents

    def _fake_completion(**kwargs):
        text = kwargs["messages"][0]["content"]

        if _is_classify_plan_prompt(text):
            return _llm_response(json.dumps({"is_multi_hop": True, "first_sub_question": _ACME_SUBQ}))

        if _is_sufficiency_prompt(text):
            is_sufficient = "Beta Inc" in text and "80 million" in text
            if is_sufficient:
                return _llm_response(json.dumps({"sufficient": True, "missing": None}))
            return _llm_response(json.dumps({"sufficient": False, "missing": "no evidence yet about Beta Inc's revenue growth"}))

        if _is_rewrite_prompt(text):
            return _llm_response(json.dumps({"next_sub_question": _BETA_SUBQ}))

        if _is_grounded_answer_prompt(text):
            chunk_ids = _CHUNK_ID_RE.findall(text)
            assert chunk_ids, f"expected chunk_ids in grounded_answer prompt:\n{text}"
            first, last = chunk_ids[0], chunk_ids[-1]
            content = (
                f"Acme Corp grew revenue 15% in fiscal year 2025. [x:1:{first}] "
                f"Beta Inc grew revenue 8% in the same period. [x:1:{last}]"
            )
            return _llm_response(content)

        return _llm_response("fiscal year 2025 revenue and growth report.")

    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)

    initial_state = RetrievalState(
        question="Compare Acme Corp's and Beta Inc's revenue growth in fiscal year 2025.",
        llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini", file_ids=None, top_k=10,
    )
    result = retrieval_graph.invoke(initial_state)
    final_state = RetrievalState.model_validate(result)

    assert final_state.is_multi_hop is True
    assert final_state.not_found is False
    assert len(set(final_state.sub_question_history)) >= 2
    assert acme_id in final_state.routed_docs
    assert beta_id in final_state.routed_docs
    assert final_state.hop_count == 2
    assert final_state.is_sufficient is True
    assert final_state.partial_answer is False
    assert len(final_state.citations) == 2


def test_hop_cap_forces_partial_answer_when_never_sufficient(monkeypatch: pytest.MonkeyPatch, two_documents) -> None:
    def _fake_completion(**kwargs):
        text = kwargs["messages"][0]["content"]

        if _is_classify_plan_prompt(text):
            return _llm_response(json.dumps({"is_multi_hop": True, "first_sub_question": _ACME_SUBQ}))

        if _is_sufficiency_prompt(text):
            # Designed to never reach sufficiency (Section 9 DoD).
            return _llm_response(json.dumps({"sufficient": False, "missing": "still missing evidence, always"}))

        if _is_rewrite_prompt(text):
            return _llm_response(json.dumps({"next_sub_question": "What else is missing about this comparison?"}))

        if _is_grounded_answer_prompt(text):
            return _llm_response("No sufficient evidence was gathered to fully answer this question.")

        return _llm_response("summary")

    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)

    initial_state = RetrievalState(
        question="Compare Acme Corp's and Beta Inc's revenue growth in fiscal year 2025.",
        llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini", file_ids=None, top_k=10,
    )
    result = retrieval_graph.invoke(initial_state)
    final_state = RetrievalState.model_validate(result)

    assert final_state.hop_count == settings.hop_cap
    assert final_state.partial_answer is True
    assert final_state.is_sufficient is False
