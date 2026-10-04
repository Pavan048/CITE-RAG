"""Thin client for MongoDB (PRD Section 6, Section 7). Every raw pymongo query lives here — nothing
in `api/`, `ingestion/`, or `retrieval/` builds a Mongo filter dict directly (code-quality rule 4).

Update methods take explicit typed optional keyword arguments rather than a generic `**fields: Any`
blob — narrower, but keeps every call site type-checked instead of passing arbitrary dict keys that
only pymongo would ever validate (code-quality rule 5).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pymongo import MongoClient

from app.config import settings
from app.models.mongo_models import DocumentRecord, JobRecord, ParentBlockRecord
from app.models.schemas import DocumentStatus, JobError, PipelineStage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoService:
    def __init__(self) -> None:
        self._client = MongoClient(settings.mongo_uri)
        self._db = self._client[settings.mongo_db_name]
        self.documents = self._db["documents"]
        self.jobs = self._db["jobs"]
        self.parent_blocks = self._db["parent_blocks"]

    def ensure_indexes(self) -> None:
        self.documents.create_index("content_hash")
        self.parent_blocks.create_index("doc_id")
        self.jobs.create_index("doc_id")

    # --- documents ---

    def insert_document(self, record: DocumentRecord) -> None:
        self.documents.insert_one(record.model_dump(by_alias=True, mode="json"))

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        doc = self.documents.find_one({"_id": doc_id})
        return DocumentRecord.model_validate(doc) if doc else None

    def find_ready_document_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        """Backs the content-hash dedup path (Section 2 Stage 5, Section 8): re-uploading a
        byte-identical file returns the existing ready doc_id instead of doing any new work."""
        doc = self.documents.find_one({"content_hash": content_hash, "status": DocumentStatus.READY.value})
        return DocumentRecord.model_validate(doc) if doc else None

    def list_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord.model_validate(d) for d in self.documents.find()]

    def update_document(
        self,
        doc_id: str,
        *,
        status: DocumentStatus | None = None,
        page_count: int | None = None,
        pages_failed: list[int] | None = None,
        content_hash: str | None = None,
    ) -> None:
        update: dict[str, object] = {"updated_at": _utcnow()}
        if status is not None:
            update["status"] = status.value
        if page_count is not None:
            update["page_count"] = page_count
        if pages_failed is not None:
            update["pages_failed"] = pages_failed
        if content_hash is not None:
            update["content_hash"] = content_hash
        self.documents.update_one({"_id": doc_id}, {"$set": update})

    def delete_document(self, doc_id: str) -> None:
        self.documents.delete_one({"_id": doc_id})
        self.parent_blocks.delete_many({"doc_id": doc_id})
        self.jobs.delete_many({"doc_id": doc_id})

    # --- jobs ---

    def insert_job(self, record: JobRecord) -> None:
        self.jobs.insert_one(record.model_dump(by_alias=True, mode="json"))

    def get_job(self, job_id: str) -> JobRecord | None:
        job = self.jobs.find_one({"_id": job_id})
        return JobRecord.model_validate(job) if job else None

    def find_stranded_jobs(self) -> list[JobRecord]:
        """Jobs left in a non-terminal status — still `queued` or actively `processing` when the
        server last stopped. The in-memory ingestion queue (ingestion/pipeline.py) doesn't survive
        a restart, so nothing will ever pick these back up on its own; see
        `pipeline.recover_stranded_jobs()` for what happens to them instead."""
        cursor = self.jobs.find({"status": {"$in": [DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value]}})
        return [JobRecord.model_validate(job) for job in cursor]

    def update_job(
        self,
        job_id: str,
        *,
        status: DocumentStatus | None = None,
        stage: PipelineStage | None = None,
        progress_pct: int | None = None,
        error: JobError | None = None,
    ) -> None:
        update: dict[str, object] = {"updated_at": _utcnow()}
        if status is not None:
            update["status"] = status.value
        if stage is not None:
            update["stage"] = stage.value
        if progress_pct is not None:
            update["progress_pct"] = progress_pct
        if error is not None:
            update["error"] = error.model_dump(mode="json")
        self.jobs.update_one({"_id": job_id}, {"$set": update})

    # --- parent_blocks ---

    def insert_parent_blocks(self, records: list[ParentBlockRecord]) -> None:
        if records:
            self.parent_blocks.insert_many([r.model_dump(mode="json") for r in records])

    def get_parent_blocks(self, parent_block_ids: list[str]) -> dict[str, str]:
        """Returns {parent_block_id: text}, used by context_budget_node to swap a surviving chunk
        for its full parent block."""
        cursor = self.parent_blocks.find({"parent_block_id": {"$in": parent_block_ids}})
        return {doc["parent_block_id"]: doc["text"] for doc in cursor}


mongo_service = MongoService()
