"""Storage-facing shapes: MongoDB documents and Qdrant payloads (PRD Section 6, Section 7).

Deliberately separate from `models/schemas.py`. These models are allowed to carry fields an API
response never should (e.g. internal ids, `pages_failed`, raw content_hash) and, conversely, never
carry request-scoped secrets (`X-LLM-Key` is never written here — see Section 8).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.models.schemas import DocumentStatus, JobError, PipelineStage


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_CAPTION = "image_caption"


# --- MongoDB: `documents` collection ---
class DocumentRecord(BaseModel):
    id: str = Field(alias="_id")
    filename: str
    page_count: int
    status: DocumentStatus
    uploaded_at: datetime
    updated_at: datetime
    pages_failed: list[int] = Field(default_factory=list)
    content_hash: str

    model_config = {"populate_by_name": True}


# --- MongoDB: `jobs` collection ---
class JobRecord(BaseModel):
    id: str = Field(alias="_id")
    doc_id: str
    status: DocumentStatus
    stage: PipelineStage
    progress_pct: int = Field(ge=0, le=100, default=0)
    error: JobError | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"populate_by_name": True}


# --- MongoDB: `parent_blocks` collection ---
class ParentBlockRecord(BaseModel):
    parent_block_id: str
    doc_id: str
    text: str


# --- Qdrant: `chunks_collection` payload ---
class ChunkPayload(BaseModel):
    doc_id: str
    page_number: int
    section_title: str | None
    content_type: ContentType
    chunk_index: int
    parent_block_id: str
    content_hash: str
    token_count: int
    # Not in PRD Section 6's literal field list, but required for the pipeline to function at all:
    # without the chunk's own text stored somewhere retrievable, rerank_node has nothing to hand
    # the reranker and generate_cite_node has nothing to build a citation `snippet` from. Storing
    # it directly on the payload (rather than a second Mongo lookup per chunk) keeps one round trip
    # per retrieval instead of two.
    chunk_text: str


# --- Qdrant: `documents_collection` payload (doc-summary vector) ---
class DocumentSummaryPayload(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    uploaded_at: datetime
    content_hash: str
