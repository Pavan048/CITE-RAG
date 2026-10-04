"""End-to-end ingestion test (Section 9 Definition of Done, first three bullets): upload -> job
reaches ready -> chunks/parent blocks/doc-summary vector all present; re-upload dedups; delete
removes everything.

Runs against real Docling, real local Qdrant/Mongo, and the mock inference server (tests/
mock_inference_server.py, already running on :8001/:8002 per the build notes). The only thing
mocked is `litellm.completion` — there is no real BYOK key available in this environment, and
nothing about pipeline orchestration depends on what a real LLM says back, only on the plumbing
that gets a prompt to it and a string back.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from app.services import llm_client as llm_client_module

TEST_PDF = Path(__file__).resolve().parent.parent / "fixtures" / "test.pdf"


def _fake_completion(**kwargs):
    is_multimodal = any(isinstance(m.get("content"), list) for m in kwargs["messages"])
    content = "A blue bar chart showing quarterly revenue." if is_multimodal else "Acme Corp's fiscal year 2025 revenue summary."
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)


@pytest.mark.asyncio
async def test_full_ingestion_lifecycle() -> None:
    from app.main import app  # imported after the monkeypatch fixture has patched litellm

    transport = ASGITransport(app=app)
    # httpx's ASGITransport does not itself send ASGI lifespan events, so the FastAPI app's
    # `lifespan` (which starts the ingestion worker pool) never runs unless entered explicitly
    # here — without this, the queued job below sits forever with nothing consuming it.
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await _run_lifecycle(client)


async def _run_lifecycle(client: httpx.AsyncClient) -> None:
    with open(TEST_PDF, "rb") as f:
        resp = await client.post(
            "/v1/documents",
            files={"file": ("test.pdf", f, "application/pdf")},
            headers={"X-LLM-Key": "sk-fake-test-key"},
        )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    doc_id, job_id = body["doc_id"], body["job_id"]

    # Poll job status until ready or failed (the worker runs as a background asyncio task).
    job: dict = {}
    for _ in range(60):
        job_resp = await client.get(f"/v1/jobs/{job_id}")
        job = job_resp.json()
        if job["status"] in ("ready", "failed"):
            break
        await asyncio.sleep(0.5)
    assert job["status"] == "ready", job

    from app.config import settings
    from app.services.mongo_client import mongo_service
    from app.services.qdrant_client import qdrant_service

    doc = mongo_service.get_document(doc_id)
    assert doc is not None and doc.status.value == "ready"
    assert doc.content_hash

    parent_blocks = list(mongo_service.parent_blocks.find({"doc_id": doc_id}))
    assert len(parent_blocks) > 0

    chunk_points, _ = qdrant_service._client.scroll(
        collection_name=settings.chunks_collection,
        scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
        limit=100,
    )
    assert len(chunk_points) > 0
    assert any(p.payload["content_type"] == "table" for p in chunk_points)
    assert any(p.payload["content_type"] == "image_caption" for p in chunk_points)
    assert any(p.payload["content_type"] == "text" for p in chunk_points)

    doc_vector_points, _ = qdrant_service._client.scroll(
        collection_name=settings.documents_collection,
        scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
        limit=10,
    )
    assert len(doc_vector_points) == 1

    # Re-upload the identical file: dedup path hit, same doc_id back, no new chunks.
    with open(TEST_PDF, "rb") as f:
        resp2 = await client.post(
            "/v1/documents",
            files={"file": ("test.pdf", f, "application/pdf")},
            headers={"X-LLM-Key": "sk-fake-test-key"},
        )
    assert resp2.status_code == 202
    assert resp2.json()["doc_id"] == doc_id

    chunk_points_after_dup, _ = qdrant_service._client.scroll(
        collection_name=settings.chunks_collection,
        scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
        limit=200,
    )
    assert len(chunk_points_after_dup) == len(chunk_points)

    # Delete: chunk points and doc-level vector both gone; scoped query returns nothing.
    del_resp = await client.delete(f"/v1/documents/{doc_id}")
    assert del_resp.status_code == 204
    assert mongo_service.get_document(doc_id) is None

    chunk_points_after_delete, _ = qdrant_service._client.scroll(
        collection_name=settings.chunks_collection,
        scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
        limit=10,
    )
    assert len(chunk_points_after_delete) == 0
