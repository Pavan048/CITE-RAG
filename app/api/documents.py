"""PRD Section 4: POST /v1/documents, GET /v1/documents, DELETE /v1/documents/{doc_id}.

Stage 1 (validate) runs entirely here, synchronously, before anything is queued — virus scan, size
check, and the fast header-only page-count check all happen in this request handler, exactly as
Section 2 Stage 1 specifies ("reject before queuing... never parse synchronously in the request
path" refers to the *Docling parse*, Stage 2 onward, not these fast checks).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.config import settings
from app.ingestion.parse import FileTooLargeError, InvalidPdfError, TooManyPagesError, validate_upload
from app.ingestion.pipeline import IngestionJob, QueueFullError, enqueue_ingestion_job
from app.models.mongo_models import DocumentRecord, JobRecord
from app.models.schemas import DocumentListItem, DocumentStatus, JobError, PipelineStage, UploadDocumentResponse
from app.services.mongo_client import mongo_service
from app.services.qdrant_client import qdrant_service
from app.services.virus_scan_service import InfectedFileError, virus_scan_service

router = APIRouter()

_UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.post("/v1/documents", status_code=202, response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    x_llm_key: str = Header(..., alias="X-LLM-Key"),
    x_llm_model: str | None = Header(None, alias="X-LLM-Model"),
) -> UploadDocumentResponse:
    # `x_llm_key` is used only to construct the IngestionJob below, passed through to the one
    # LLM call site that needs it (Section 8) — it is never written into a log line, a Mongo
    # document, or this function's own error paths.
    llm_model = x_llm_model or settings.default_llm_model

    os.makedirs(settings.upload_dir, exist_ok=True)
    temp_path = os.path.join(settings.upload_dir, f"upload-{uuid.uuid4()}.pdf")
    hasher = hashlib.sha256()
    size = 0
    # Streamed to disk in bounded chunks, never buffered whole in memory (Section 8).
    with open(temp_path, "wb") as f:
        while True:
            data = await file.read(_UPLOAD_CHUNK_BYTES)
            if not data:
                break
            hasher.update(data)
            size += len(data)
            f.write(data)
    content_hash = hasher.hexdigest()

    try:
        virus_scan_service.scan_file(temp_path)
        page_count = validate_upload(temp_path, size)
    except InfectedFileError as e:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileTooLargeError as e:
        os.remove(temp_path)
        raise HTTPException(status_code=413, detail=str(e)) from e
    except (InvalidPdfError, TooManyPagesError) as e:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Content-hash dedup (Section 2 Stage 5, Section 8): re-uploading a byte-identical file
    # returns the SAME existing doc_id with an already-`ready` job — this literally is that
    # document already, so there is nothing new to parse, embed, or index.
    existing = mongo_service.find_ready_document_by_content_hash(content_hash)
    if existing is not None:
        os.remove(temp_path)
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        mongo_service.insert_job(
            JobRecord(_id=job_id, doc_id=existing.id, status=DocumentStatus.READY, stage=PipelineStage.READY, progress_pct=100, created_at=now, updated_at=now)
        )
        return UploadDocumentResponse(doc_id=existing.id, job_id=job_id)

    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    final_path = os.path.join(settings.upload_dir, f"{doc_id}.pdf")
    os.replace(temp_path, final_path)

    mongo_service.insert_document(
        DocumentRecord(
            _id=doc_id, filename=file.filename or "document.pdf", page_count=page_count,
            status=DocumentStatus.QUEUED, uploaded_at=now, updated_at=now, pages_failed=[], content_hash=content_hash,
        )
    )
    mongo_service.insert_job(
        JobRecord(_id=job_id, doc_id=doc_id, status=DocumentStatus.QUEUED, stage=PipelineStage.VALIDATE, progress_pct=0, created_at=now, updated_at=now)
    )

    try:
        enqueue_ingestion_job(
            IngestionJob(
                doc_id=doc_id, job_id=job_id, file_path=final_path, filename=file.filename or "document.pdf",
                content_hash=content_hash, llm_api_key=x_llm_key, llm_model=llm_model,
            )
        )
    except QueueFullError as e:
        mongo_service.update_document(doc_id, status=DocumentStatus.FAILED)
        mongo_service.update_job(job_id, status=DocumentStatus.FAILED, error=JobError(stage=PipelineStage.VALIDATE, message=str(e)))
        raise HTTPException(status_code=503, detail=str(e)) from e

    return UploadDocumentResponse(doc_id=doc_id, job_id=job_id)


@router.get("/v1/documents", response_model=list[DocumentListItem])
async def list_documents() -> list[DocumentListItem]:
    return [
        DocumentListItem(doc_id=d.id, filename=d.filename, page_count=d.page_count, status=d.status, uploaded_at=d.uploaded_at)
        for d in mongo_service.list_documents()
    ]


@router.delete("/v1/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str) -> None:
    qdrant_service.delete_document(doc_id)
    mongo_service.delete_document(doc_id)
