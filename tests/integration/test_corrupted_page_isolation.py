"""Section 9 Definition of Done: "A corrupted single page inside an otherwise valid PDF does not
fail the whole ingestion job."

Docling's own per-page fault isolation is real (confirmed by reading docling's base_pipeline.py
during the build — see docling_service.py's module docstring) but relies on encountering an
actually-malformed page, which is fiddly and environment-fragile to construct by hand-corrupting
real PDF bytes (PDF renderers tend to be very forgiving of garbage content streams, so a hand-
crafted corruption might simply get silently ignored rather than reproducing the failure mode this
test needs). Instead, this exercises OUR handling of that outcome directly: `docling_service.
convert` is made to return a `PARTIAL_SUCCESS` result with one failed page, and the full pipeline
is run through the real API to confirm the job still reaches `ready` (not `failed`) with
`pages_failed` correctly surfaced — Docling's own internal isolation is not what's under test here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from app.services import docling_service as docling_service_module
from app.services import llm_client as llm_client_module
from app.services.docling_service import DoclingConversionStatus, DoclingConvertResult

TEST_PDF = Path(__file__).resolve().parent.parent / "fixtures" / "test.pdf"


def _fake_completion(**kwargs):
    is_multimodal = any(isinstance(m.get("content"), list) for m in kwargs["messages"])
    content = "A blue bar chart." if is_multimodal else "A fiscal year 2025 revenue report."
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)


@pytest.fixture()
def _partial_success_docling(monkeypatch: pytest.MonkeyPatch):
    real_convert = docling_service_module.docling_service.convert

    def _patched_convert(file_path: str) -> DoclingConvertResult:
        real_result = real_convert(file_path)
        return DoclingConvertResult(
            document=real_result.document,
            status=DoclingConversionStatus.PARTIAL_SUCCESS,
            page_count=real_result.page_count,
            failed_page_numbers=[2],
        )

    monkeypatch.setattr(docling_service_module.docling_service, "convert", _patched_convert)


@pytest.mark.asyncio
async def test_partial_success_does_not_fail_the_job(_partial_success_docling) -> None:
    from app.main import app

    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with open(TEST_PDF, "rb") as f:
                resp = await client.post(
                    "/v1/documents", files={"file": ("test.pdf", f, "application/pdf")}, headers={"X-LLM-Key": "sk-fake"}
                )
            assert resp.status_code == 202, resp.text
            doc_id, job_id = resp.json()["doc_id"], resp.json()["job_id"]

            job: dict = {}
            for _ in range(60):
                job = (await client.get(f"/v1/jobs/{job_id}")).json()
                if job["status"] in ("ready", "failed"):
                    break
                await asyncio.sleep(0.5)

            assert job["status"] == "ready", job
            assert job["pages_failed"] == [2]

            await client.delete(f"/v1/documents/{doc_id}")
