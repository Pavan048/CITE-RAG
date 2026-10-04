"""API request/response shapes (PRD Section 4, Section 7).

Intentionally decoupled from `mongo_models.py`: nothing here may carry a storage-only or
key-related field. In particular `X-LLM-Key` must never appear on any model in this file — it is
read as a request header and passed as a plain function argument, never modeled as a body field.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class PipelineStage(str, Enum):
    VALIDATE = "validate"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    READY = "ready"


class JobError(BaseModel):
    stage: PipelineStage
    message: str


class UploadDocumentResponse(BaseModel):
    doc_id: str
    job_id: str


class JobStatusResponse(BaseModel):
    status: DocumentStatus
    stage: PipelineStage
    progress_pct: int = Field(ge=0, le=100)
    error: JobError | None = None
    pages_failed: list[int] | None = None


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    status: DocumentStatus
    uploaded_at: datetime


class QueryRequest(BaseModel):
    question: str
    file_ids: list[str] | None = None
    top_k: int | None = None


class Citation(BaseModel):
    doc_id: str
    page_number: int
    chunk_id: str
    snippet: str


# --- Debug trace: not part of the PRD's core contract (Section 4 doesn't mention it) — an
# additive, optional field so a caller following the literal documented contract sees nothing new,
# while the UI's debug panel can show exactly which nodes ran, what was retrieved at each hop, and
# every prompt/response, instead of that only being visible in server logs.


class ChunkTrace(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    content_type: str
    score: float
    text_preview: str


class HopTrace(BaseModel):
    hop_number: int
    sub_question: str
    chunks_retrieved: list[ChunkTrace]


class LLMCallTrace(BaseModel):
    node: str
    prompt: str
    response: str


class DebugInfo(BaseModel):
    graph_path: list[str]
    is_multi_hop: bool
    hop_count: int
    not_found: bool
    sub_question_history: list[str]
    hops: list[HopTrace]
    reranked_chunks: list[ChunkTrace]
    context_block_count: int
    llm_calls: list[LLMCallTrace]


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    routed_docs: list[str] | None = None
    partial_answer: bool
    debug: DebugInfo | None = None
