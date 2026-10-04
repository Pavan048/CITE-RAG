"""Smoke test against the local system MongoDB (localhost:27017), scoped entirely to the
`rag_kb` database so it never touches any other data on that server."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models.mongo_models import DocumentRecord, JobRecord, ParentBlockRecord
from app.models.schemas import DocumentStatus, JobError, PipelineStage
from app.services.mongo_client import MongoService


def test_mongo_roundtrip() -> None:
    svc = MongoService()
    svc.ensure_indexes()

    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    svc.insert_document(
        DocumentRecord(
            _id=doc_id, filename="test.pdf", page_count=2, status=DocumentStatus.QUEUED,
            uploaded_at=now, updated_at=now, pages_failed=[], content_hash="hash123",
        )
    )
    svc.insert_job(
        JobRecord(
            _id=job_id, doc_id=doc_id, status=DocumentStatus.QUEUED, stage=PipelineStage.VALIDATE,
            progress_pct=0, created_at=now, updated_at=now,
        )
    )

    fetched = svc.get_document(doc_id)
    assert fetched is not None and fetched.filename == "test.pdf"

    svc.update_document(doc_id, status=DocumentStatus.READY, pages_failed=[2])
    assert svc.get_document(doc_id).status == DocumentStatus.READY
    assert svc.get_document(doc_id).pages_failed == [2]

    svc.update_job(job_id, status=DocumentStatus.FAILED, stage=PipelineStage.PARSE, error=JobError(stage=PipelineStage.PARSE, message="boom"))
    job = svc.get_job(job_id)
    assert job.status == DocumentStatus.FAILED
    assert job.error.message == "boom"

    dup = svc.find_ready_document_by_content_hash("hash123")
    assert dup is not None and dup.id == doc_id

    svc.insert_parent_blocks([ParentBlockRecord(parent_block_id="pb1", doc_id=doc_id, text="full section text")])
    blocks = svc.get_parent_blocks(["pb1"])
    assert blocks["pb1"] == "full section text"

    svc.delete_document(doc_id)
    assert svc.get_document(doc_id) is None
    assert svc.get_parent_blocks(["pb1"]) == {}
