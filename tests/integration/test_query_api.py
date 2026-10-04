"""Validates POST /v1/query end-to-end through the actual HTTP layer (build order step 7: "api/
routers wiring it all together") — distinct from test_retrieval_single_hop.py, which invokes the
compiled graph directly and so never exercises FastAPI's header parsing or QueryRequest/
QueryResponse (de)serialization.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from app.services import llm_client as llm_client_module
from app.services.mongo_client import mongo_service
from app.services.qdrant_client import qdrant_service

from .test_ingestion_e2e import TEST_PDF

_CHUNK_ID_IN_PROMPT_RE = re.compile(r"chunk_id=([0-9a-f-]{36})")


def _fake_completion(**kwargs):
    messages = kwargs["messages"]
    is_multimodal = any(isinstance(m.get("content"), list) for m in messages)
    text = messages[0]["content"] if not is_multimodal else messages[0]["content"][0]["text"]

    if is_multimodal:
        content = "A blue bar chart showing quarterly revenue."
    elif "is_multi_hop" in text:
        content = '{"is_multi_hop": false, "first_sub_question": null}'
    elif "citation options" in text:
        chunk_id = _CHUNK_ID_IN_PROMPT_RE.search(text).group(1)
        content = f"Acme Corp reported $120 million in revenue. [ignored:99:{chunk_id}]"
    else:
        content = "Acme Corp's fiscal year 2025 revenue summary."

    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)


@pytest.fixture(autouse=True)
def _permissive_doc_routing_floor(monkeypatch: pytest.MonkeyPatch):
    from app.config import settings

    monkeypatch.setattr(settings, "doc_routing_similarity_floor", -1.0)


@pytest.mark.asyncio
async def test_query_endpoint_returns_valid_response_shape() -> None:
    from app.main import app

    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with open(TEST_PDF, "rb") as f:
                upload_resp = await client.post(
                    "/v1/documents", files={"file": ("acme.pdf", f, "application/pdf")}, headers={"X-LLM-Key": "sk-fake"}
                )
            doc_id, job_id = upload_resp.json()["doc_id"], upload_resp.json()["job_id"]

            import asyncio

            for _ in range(60):
                job = (await client.get(f"/v1/jobs/{job_id}")).json()
                if job["status"] in ("ready", "failed"):
                    break
                await asyncio.sleep(0.5)
            assert job["status"] == "ready", job

            try:
                resp = await client.post(
                    "/v1/query",
                    json={"question": "What was Acme Corp's revenue in fiscal year 2025?"},
                    headers={"X-LLM-Key": "sk-fake"},
                )
                assert resp.status_code == 200, resp.text
                body = resp.json()

                assert "$120 million" in body["answer"]
                assert body["partial_answer"] is False
                assert isinstance(body["citations"], list) and len(body["citations"]) == 1
                citation = body["citations"][0]
                assert set(citation.keys()) == {"doc_id", "page_number", "chunk_id", "snippet"}
                assert citation["doc_id"] == doc_id
                assert doc_id in body["routed_docs"]

                # Scoped query (file_ids pinned): routing skipped, routed_docs must be null.
                scoped_resp = await client.post(
                    "/v1/query",
                    json={"question": "What was the revenue?", "file_ids": [doc_id]},
                    headers={"X-LLM-Key": "sk-fake"},
                )
                assert scoped_resp.json()["routed_docs"] is None
            finally:
                await client.delete(f"/v1/documents/{doc_id}")

            # Section 9 DoD: a query scoped to a just-deleted doc_id returns "not found," not
            # stale results — file_ids is pinned so routing is skipped, hybrid search against the
            # now-nonexistent doc_id's chunks comes back empty, and generate_cite_node's empty-
            # context fallback produces the abstention answer with zero citations.
            after_delete_resp = await client.post(
                "/v1/query",
                json={"question": "What was the revenue?", "file_ids": [doc_id]},
                headers={"X-LLM-Key": "sk-fake"},
            )
            after_delete_body = after_delete_resp.json()
            assert after_delete_body["citations"] == []
            assert "could not find" in after_delete_body["answer"].lower()

    assert mongo_service.get_document(doc_id) is None
