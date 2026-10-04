"""Validates recover_stranded_jobs() (app/ingestion/pipeline.py) — added after a review flagged
that a job stuck in `queued`/`processing` when the process stops is silently stranded forever,
since the in-memory ingestion queue doesn't survive a restart. This can only fail such jobs
cleanly, never resume them: `llm_api_key` is never persisted (Section 8), so there is no key left
to finish an interrupted job with.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.ingestion.pipeline import recover_stranded_jobs
from app.models.mongo_models import DocumentRecord, JobRecord
from app.models.schemas import DocumentStatus, PipelineStage
from app.services.mongo_client import mongo_service


def _insert_stranded(status: DocumentStatus, stage: PipelineStage) -> tuple[str, str]:
    doc_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    mongo_service.insert_document(
        DocumentRecord(_id=doc_id, filename="stranded.pdf", page_count=1, status=status, uploaded_at=now, updated_at=now, pages_failed=[], content_hash=f"hash-{doc_id}")
    )
    mongo_service.insert_job(
        JobRecord(_id=job_id, doc_id=doc_id, status=status, stage=stage, progress_pct=40, created_at=now, updated_at=now)
    )
    return doc_id, job_id


def test_recover_stranded_jobs_fails_queued_and_processing_jobs_cleanly() -> None:
    queued_doc_id, queued_job_id = _insert_stranded(DocumentStatus.QUEUED, PipelineStage.VALIDATE)
    processing_doc_id, processing_job_id = _insert_stranded(DocumentStatus.PROCESSING, PipelineStage.EMBED)
    ready_doc_id, ready_job_id = _insert_stranded(DocumentStatus.READY, PipelineStage.READY)  # must be left alone

    try:
        recover_stranded_jobs()

        queued_job = mongo_service.get_job(queued_job_id)
        assert queued_job.status == DocumentStatus.FAILED
        assert queued_job.stage == PipelineStage.VALIDATE  # stage is preserved, not reset
        assert "cannot be resumed automatically" in queued_job.error.message
        assert mongo_service.get_document(queued_doc_id).status == DocumentStatus.FAILED

        processing_job = mongo_service.get_job(processing_job_id)
        assert processing_job.status == DocumentStatus.FAILED
        assert processing_job.stage == PipelineStage.EMBED
        assert mongo_service.get_document(processing_doc_id).status == DocumentStatus.FAILED

        # A job that had already finished before the restart must not be touched.
        ready_job = mongo_service.get_job(ready_job_id)
        assert ready_job.status == DocumentStatus.READY
        assert ready_job.error is None
        assert mongo_service.get_document(ready_doc_id).status == DocumentStatus.READY

        # Idempotent: running it again with nothing left stranded is a no-op, not an error.
        recover_stranded_jobs()
        assert mongo_service.get_job(queued_job_id).status == DocumentStatus.FAILED
    finally:
        for doc_id in (queued_doc_id, processing_doc_id, ready_doc_id):
            mongo_service.delete_document(doc_id)
