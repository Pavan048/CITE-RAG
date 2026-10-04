"""Validates the streaming query endpoint (POST /v1/query/stream) and the debug trace attached to
both query endpoints. Neither is part of the PRD's documented contract (Section 4) — both were
added on top of the working, tested retrieval graph, so this exists to prove the addition didn't
regress anything and that the new pieces (graph_path, per-hop chunk counts, streamed citation
resolution) actually work.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from app.ingestion.chunk import chunk_document
from app.ingestion.embed import embed_chunks, embed_document_summary
from app.ingestion.index import index_document
from app.ingestion.parse import parse_document
from app.services import llm_client as llm_client_module
from app.services.mongo_client import mongo_service
from app.services.qdrant_client import qdrant_service

from .test_ingestion_e2e import TEST_PDF

_CHUNK_ID_IN_PROMPT_RE = re.compile(r"chunk_id=([0-9a-f-]{36})")


def _parse_sse_events(raw_lines: list[str]) -> list[tuple[str, dict]]:
    # SSE pairs an "event: <name>" line with the following "data: <json>" line — parsed
    # explicitly rather than pattern-matching the JSON payload text, which can coincidentally
    # contain a substring like `"text"` in an unrelated field (e.g. the debug trace's
    # `content_type: "text"`) and silently misclassify a "done" event as a "token".
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    for line in raw_lines:
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:") and current_event is not None:
            events.append((current_event, json.loads(line[len("data:"):].strip())))
            current_event = None
    return events


def _fake_completion(**kwargs):
    text = kwargs["messages"][0]["content"]
    is_multimodal = any(isinstance(m.get("content"), list) for m in kwargs["messages"])
    content = "A blue bar chart." if is_multimodal else '{"is_multi_hop": false, "first_sub_question": null}' if "is_multi_hop" in text else "summary"
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def _fake_stream_text(*, prompt: str, api_key: str, model: str):
    chunk_id = _CHUNK_ID_IN_PROMPT_RE.search(prompt).group(1)
    # Split the citation marker itself across chunk boundaries on purpose — proves resolution only
    # ever happens against the fully accumulated text, never mid-stream.
    for piece in ["Acme Corp reported $120 million in revenue. [", f"ignored:1:{chunk_id}", "]"]:
        yield piece


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)
    monkeypatch.setattr(llm_client_module.llm_client, "stream_text", _fake_stream_text)


@pytest.fixture(autouse=True)
def _permissive_doc_routing_floor(monkeypatch: pytest.MonkeyPatch):
    from app.config import settings

    monkeypatch.setattr(settings, "doc_routing_similarity_floor", -1.0)


def _ingest_test_pdf() -> str:
    doc_id = str(uuid.uuid4())
    parsed = parse_document(str(TEST_PDF), "test.pdf", llm_api_key="sk-fake", llm_model="openai/gpt-4o-mini")
    chunks, parent_blocks = chunk_document(parsed, doc_id)
    mongo_service.insert_parent_blocks(parent_blocks)
    chunk_embeddings = embed_chunks([p.chunk_text for _, p in chunks])
    summary_dense = embed_document_summary("Acme Corp fiscal year 2025 revenue report.")
    index_document(doc_id, chunks, chunk_embeddings, summary_dense, "test.pdf", parsed.page_count, datetime.now(timezone.utc), "hash-stream-debug-test")
    return doc_id


@pytest.mark.asyncio
async def test_query_stream_sends_tokens_then_resolved_done_event() -> None:
    from app.main import app

    doc_id = _ingest_test_pdf()
    try:
        transport = ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
                async with client.stream(
                    "POST",
                    "/v1/query/stream",
                    json={"question": "What was Acme Corp's revenue in fiscal year 2025?"},
                    headers={"X-LLM-Key": "sk-fake"},
                ) as response:
                    assert response.status_code == 200
                    raw_lines = [line async for line in response.aiter_lines()]

        events = _parse_sse_events(raw_lines)
        token_events = [payload for name, payload in events if name == "token"]
        done_payload = next(payload for name, payload in events if name == "done")

        assert len(token_events) == 3  # the three pieces _fake_stream_text yielded, unmerged
        assert "$120 million" in done_payload["answer"]
        assert len(done_payload["citations"]) == 1
        assert done_payload["citations"][0]["doc_id"] == doc_id  # re-derived from trusted data, not the "ignored:" the fake model wrote

        debug = done_payload["debug"]
        assert debug["graph_path"][0] == "classify_plan"
        assert "generate_cite" in debug["graph_path"]
        assert debug["is_multi_hop"] is False
        assert debug["hop_count"] == 0
        assert len(debug["hops"]) == 1
        assert debug["hops"][0]["chunks_retrieved"], "expected at least one retrieved chunk in the debug trace"
        assert any(call["node"] == "classify_plan" for call in debug["llm_calls"])
    finally:
        qdrant_service.delete_document(doc_id)
        mongo_service.parent_blocks.delete_many({"doc_id": doc_id})


@pytest.mark.asyncio
async def test_non_streaming_query_also_carries_debug_info() -> None:
    from app.main import app

    doc_id = _ingest_test_pdf()
    try:
        transport = ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/query",
                    json={"question": "What was Acme Corp's revenue in fiscal year 2025?"},
                    headers={"X-LLM-Key": "sk-fake"},
                )
        assert resp.status_code == 200
        body = resp.json()
        debug = body["debug"]
        assert debug["graph_path"] == ["classify_plan", "route_retrieve", "accumulate", "rerank", "context_budget", "generate_cite"]
        assert debug["reranked_chunks"]
        assert debug["context_block_count"] >= 1
    finally:
        qdrant_service.delete_document(doc_id)
        mongo_service.parent_blocks.delete_many({"doc_id": doc_id})


@pytest.mark.asyncio
async def test_query_stream_yields_clean_error_event_on_pre_generation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for a real bug found live: an invalid API key raised inside the
    *pre-generation* graph phase (classify_plan's LLM call) used to escape event_stream()'s
    try/except entirely — only the final streaming call was wrapped — so the ASGI connection was
    aborted mid-response instead of sending a clean `event: error` frame. A browser reports an
    abruptly-closed connection as a bare "network error" with no useful detail, which is exactly
    what surfaced this. See api/query.py's query_stream for the fix (the try/except now wraps the
    entire body).
    """
    from app.main import app

    def _broken_completion(**kwargs):
        # Mirrors litellm's real behavior (confirmed live during development): the exception
        # message echoes back the actual key it was called with.
        raise RuntimeError(f"Incorrect API key provided: {kwargs['api_key']}")

    monkeypatch.setattr(llm_client_module.litellm, "completion", _broken_completion)

    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
            async with client.stream(
                "POST",
                "/v1/query/stream",
                json={"question": "What was Acme Corp's revenue in fiscal year 2025?"},
                headers={"X-LLM-Key": "sk-fake"},
            ) as response:
                # The connection itself must still succeed (200) — the failure is communicated as
                # an SSE frame within an otherwise-normal response, not an aborted connection.
                assert response.status_code == 200
                raw_lines = [line async for line in response.aiter_lines()]

    events = _parse_sse_events(raw_lines)
    assert len(events) == 1
    event_name, payload = events[0]
    assert event_name == "error"
    assert "Incorrect API key" in payload["message"]
    assert "sk-fake" not in payload["message"]  # llm_client already redacted the key before raising
