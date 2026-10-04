"""PRD Section 2 orchestrator, and — per the user's "use FastAPI only" instruction — the async job
runner. There is no Celery/RQ broker; a bounded in-process `asyncio.Queue` plus a fixed pool of
worker coroutines (started in the FastAPI app's lifespan) plays the same role: `enqueue()` never
blocks the request path (Section 2 Stage 1: "never parse synchronously in the request path"), and
a full queue is rejected outright rather than awaited, which is what "backpressure via queue depth,
not unbounded thread spawning" (Section 8) means for a request handler — a slow drain should push
back on new uploads, not hang the caller.

Because everything runs in one process, `llm_api_key` (Section 8) never has to cross a broker or
any persistence boundary to reach the worker — it stays a plain Python argument from the API
request handler down to the one `llm_client` call that uses it, and is never written to Mongo, a
log line, or an error payload alongside it (see the `except` block below).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from pydantic import BaseModel

from app.config import settings
from app.ingestion.chunk import chunk_document, generate_document_summary
from app.ingestion.embed import embed_chunks, embed_document_summary
from app.ingestion.index import index_document
from app.ingestion.parse import parse_document
from app.models.schemas import DocumentStatus, JobError, PipelineStage
from app.services.mongo_client import mongo_service


class IngestionJob(BaseModel):
    doc_id: str
    job_id: str
    file_path: str
    filename: str
    content_hash: str
    llm_api_key: str
    llm_model: str


class QueueFullError(Exception):
    pass


_queue: asyncio.Queue[IngestionJob] = asyncio.Queue(maxsize=settings.ingestion_queue_maxsize)
_workers: list[asyncio.Task] = []


def enqueue_ingestion_job(job: IngestionJob) -> None:
    try:
        _queue.put_nowait(job)
    except asyncio.QueueFull as e:
        raise QueueFullError(f"Ingestion queue is at capacity ({settings.ingestion_queue_maxsize})") from e


async def _worker_loop() -> None:
    while True:
        job = await _queue.get()
        try:
            await run_ingestion_pipeline(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            # run_ingestion_pipeline already records the failure on the job/document; a worker
            # loop must never die from one bad job, or every job queued behind it stalls forever.
            pass
        finally:
            _queue.task_done()


def recover_stranded_jobs() -> None:
    """Called once at startup (main.py's lifespan), before `start_workers()`. A job left `queued`
    or `processing` when the process last stopped is stranded: the in-memory `asyncio.Queue` that
    held it is gone, so no fresh worker will ever pick it back up, and its Mongo record would
    otherwise sit in that state forever with no indication anything is wrong.

    This cannot resume the job — `llm_api_key` is never persisted, anywhere (Section 8), so there
    is no key left to finish parsing/captioning/summarizing with even if we wanted to resume where
    it left off. The correct, honest fix is to fail it cleanly with an actionable message, not to
    silently leave it stuck or pretend a resume is possible when it structurally isn't.
    """
    for job in mongo_service.find_stranded_jobs():
        error = JobError(
            stage=job.stage,
            message="Ingestion was interrupted by a server restart and cannot be resumed automatically (the BYOK key is never persisted) — please re-upload the file.",
        )
        mongo_service.update_job(job.id, status=DocumentStatus.FAILED, error=error)
        mongo_service.update_document(job.doc_id, status=DocumentStatus.FAILED)


def start_workers() -> None:
    for _ in range(settings.max_concurrent_jobs):
        _workers.append(asyncio.create_task(_worker_loop()))


async def stop_workers() -> None:
    for w in _workers:
        w.cancel()
    await asyncio.gather(*_workers, return_exceptions=True)
    _workers.clear()


async def run_ingestion_pipeline(job: IngestionJob) -> None:
    stage = PipelineStage.PARSE
    try:
        mongo_service.update_document(job.doc_id, status=DocumentStatus.PROCESSING)
        mongo_service.update_job(job.job_id, status=DocumentStatus.PROCESSING, stage=PipelineStage.PARSE, progress_pct=10)

        parsed = await asyncio.to_thread(parse_document, job.file_path, job.filename, job.llm_api_key, job.llm_model)
        if parsed.pages_failed:
            mongo_service.update_document(job.doc_id, pages_failed=parsed.pages_failed)

        stage = PipelineStage.CHUNK
        mongo_service.update_job(job.job_id, stage=PipelineStage.CHUNK, progress_pct=40)
        chunks, parent_blocks = chunk_document(parsed, job.doc_id)
        summary_text = await asyncio.to_thread(generate_document_summary, parsed, job.filename, job.llm_api_key, job.llm_model)
        if parent_blocks:
            mongo_service.insert_parent_blocks(parent_blocks)

        stage = PipelineStage.EMBED
        mongo_service.update_job(job.job_id, stage=PipelineStage.EMBED, progress_pct=65)
        chunk_texts = [payload.chunk_text for _, payload in chunks]
        chunk_embeddings = await asyncio.to_thread(embed_chunks, chunk_texts)
        summary_dense = await asyncio.to_thread(embed_document_summary, summary_text)

        stage = PipelineStage.INDEX
        mongo_service.update_job(job.job_id, stage=PipelineStage.INDEX, progress_pct=90)
        await asyncio.to_thread(
            index_document, job.doc_id, chunks, chunk_embeddings, summary_dense,
            job.filename, parsed.page_count, datetime.now(timezone.utc), job.content_hash,
        )

        # Only mark ready once both Qdrant writes above and this Mongo update have succeeded
        # (Section 2 Stage 5) — a crash between index_document() and here leaves the document in
        # PROCESSING, which is correct: it is not yet safe to call this document searchable.
        mongo_service.update_document(job.doc_id, status=DocumentStatus.READY, page_count=parsed.page_count, content_hash=job.content_hash)
        mongo_service.update_job(job.job_id, status=DocumentStatus.READY, stage=PipelineStage.READY, progress_pct=100)

        if os.path.exists(job.file_path):
            os.remove(job.file_path)
    except Exception as e:
        # `str(e)` only — never include `job.llm_api_key` or the raw request in this message
        # (Section 8: the key must never appear in a Mongo document or error payload).
        mongo_service.update_document(job.doc_id, status=DocumentStatus.FAILED)
        mongo_service.update_job(job.job_id, status=DocumentStatus.FAILED, stage=stage, error=JobError(stage=stage, message=str(e)))
        raise
