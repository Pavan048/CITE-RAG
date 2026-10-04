"""PRD Section 4: GET /v1/jobs/{job_id}."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import JobStatusResponse
from app.services.mongo_client import mongo_service

router = APIRouter()


@router.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = mongo_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    document = mongo_service.get_document(job.doc_id)
    pages_failed = document.pages_failed if document and document.pages_failed else None
    return JobStatusResponse(status=job.status, stage=job.stage, progress_pct=job.progress_pct, error=job.error, pages_failed=pages_failed)
