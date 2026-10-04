"""Section 9 Definition of Done: "X-LLM-Key does not appear in any log line, Mongo document, or
error payload (grep test across the whole request/response cycle, not just the upload endpoint)."

Built after discovering, live, that this is not automatically true: a real provider's own
exception message can echo the raw key back (confirmed against a real OpenAI key during
development — an invalid-key `AuthenticationError`'s text reads "Incorrect API key provided:
<the actual key>..."). That's what `services/llm_client.py`'s redaction wrapper (`LLMCallError`)
exists to fix; this test proves the fix, not just the intent.
"""

from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone

import pytest

from app.ingestion.pipeline import IngestionJob, run_ingestion_pipeline
from app.models.mongo_models import DocumentRecord, JobRecord
from app.models.schemas import DocumentStatus, PipelineStage
from app.services import llm_client as llm_client_module
from app.services.llm_client import LLMCallError, LLMClient
from app.services.mongo_client import mongo_service

_FAKE_KEY = "sk-test-distinctive-secret-9f8e7d6c5b4a"


def _raise_with_key_embedded(**kwargs):
    # Mirrors litellm's real behavior for an invalid key: the exception text contains the key.
    raise RuntimeError(f"Incorrect API key provided: {kwargs['api_key']}. You can find your API key at ...")


def test_llm_client_redacts_key_from_raised_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module.litellm, "completion", _raise_with_key_embedded)
    client = LLMClient()

    with pytest.raises(LLMCallError) as exc_info:
        client.complete_text(prompt="hi", api_key=_FAKE_KEY, model="openai/gpt-4o-mini")
    assert _FAKE_KEY not in str(exc_info.value)
    assert _FAKE_KEY not in repr(exc_info.value)
    # `raise ... from None` sets __cause__ to None and __suppress_context__ to True — it does NOT
    # clear __context__ itself (Python still records "the exception being handled" there; only
    # the default traceback/logging formatters respect __suppress_context__ and skip printing it).
    # So the real, observable guarantee is: nothing that FORMATS this exception the normal way ever
    # shows the key — check that directly, rather than asserting __context__ is None, which
    # `from None` was never going to make true.
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    formatted = "".join(traceback.format_exception(type(exc_info.value), exc_info.value, exc_info.value.__traceback__))
    assert _FAKE_KEY not in formatted

    with pytest.raises(LLMCallError) as exc_info_img:
        client.caption_image(prompt="describe", image_bytes=b"fake", api_key=_FAKE_KEY, model="openai/gpt-4o-mini")
    assert _FAKE_KEY not in str(exc_info_img.value)


@pytest.mark.asyncio
async def test_pipeline_failure_never_writes_key_to_mongo(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(llm_client_module.litellm, "completion", _raise_with_key_embedded)

    from app.services.docling_service import docling_service

    doc_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pdf_path = tmp_path / "test.pdf"
    import shutil

    fixture = __import__("pathlib").Path(__file__).resolve().parent.parent / "fixtures" / "test.pdf"
    shutil.copy(fixture, pdf_path)

    mongo_service.insert_document(
        DocumentRecord(_id=doc_id, filename="test.pdf", page_count=2, status=DocumentStatus.QUEUED, uploaded_at=now, updated_at=now, pages_failed=[], content_hash="h")
    )
    mongo_service.insert_job(
        JobRecord(_id=job_id, doc_id=doc_id, status=DocumentStatus.QUEUED, stage=PipelineStage.VALIDATE, progress_pct=0, created_at=now, updated_at=now)
    )

    job = IngestionJob(doc_id=doc_id, job_id=job_id, file_path=str(pdf_path), filename="test.pdf", content_hash="h", llm_api_key=_FAKE_KEY, llm_model="openai/gpt-4o-mini")
    with pytest.raises(Exception):
        await run_ingestion_pipeline(job)

    failed_job = mongo_service.get_job(job_id)
    assert failed_job.status == DocumentStatus.FAILED
    assert _FAKE_KEY not in failed_job.error.message
    assert "[REDACTED]" in failed_job.error.message

    # Grep every document in every collection this app owns — the whole point of the DoD bullet
    # is that this key must not appear ANYWHERE in storage, not just in the one field we suspect.
    for collection in (mongo_service.documents, mongo_service.jobs, mongo_service.parent_blocks):
        for doc in collection.find():
            assert _FAKE_KEY not in str(doc), f"key leaked into {collection.name}: {doc}"

    mongo_service.delete_document(doc_id)
