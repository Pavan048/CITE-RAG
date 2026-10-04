"""Central, env-driven settings. Every tunable named in PRD Section 8 ("configurable limits...
all via config.py, never hardcoded inline") lives here — nothing below should be duplicated as a
literal constant inside ingestion/, retrieval/, or services/.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Storage ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    chunks_collection: str = Field(default="chunks_collection")
    documents_collection: str = Field(default="documents_collection")

    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db_name: str = Field(default="rag_kb")

    # --- Upload / ingestion limits (Section 2 Stage 1, Section 8) ---
    upload_dir: str = Field(default="/tmp/rag_uploads")
    max_file_size_mb: int = Field(default=200)
    max_page_count: int = Field(default=2000)
    max_concurrent_jobs: int = Field(default=4)
    ingestion_queue_maxsize: int = Field(default=100)
    page_parallelism: int = Field(default=4)

    # --- Virus scanning (Section 2 Stage 1) ---
    virus_scan_enabled: bool = Field(default=False)
    clamd_host: str = Field(default="localhost")
    clamd_port: int = Field(default=3310)

    # --- Chunking (Section 1, Section 2 Stage 3) ---
    chunk_target_tokens: int = Field(default=512)
    chunk_overlap_ratio: float = Field(default=0.125)  # midpoint of the 10-15% band
    doc_summary_min_tokens: int = Field(default=150)
    doc_summary_max_tokens: int = Field(default=250)
    # A 2000-page document's full extracted text can't go into one summarization prompt; this caps
    # how much of it we feed in (roughly the leading ~10k tokens' worth of characters).
    doc_summary_source_max_chars: int = Field(default=40_000)
    embedding_batch_size: int = Field(default=64)

    # --- Embedding / reranker self-hosted services (Section 1) ---
    embedding_service_url: str = Field(default="http://localhost:8001")
    embedding_dim: int = Field(default=1024)
    reranker_service_url: str = Field(default="http://localhost:8002")
    inference_service_timeout_s: float = Field(default=30.0)

    # --- Document routing / hybrid retrieval (Section 3) ---
    doc_routing_similarity_floor: float = Field(default=0.35)
    doc_routing_top_k: int = Field(default=30)
    chunk_fusion_top_k: int = Field(default=50)
    rrf_k: int = Field(default=60)
    default_top_k: int = Field(default=10)

    # --- Multi-hop loop (Section 3) ---
    hop_cap: int = Field(default=4)

    # --- Context budgeting (Section 3) ---
    # Kept well under the ~2500-token "context cliff" region reported in the Jan 2026 arXiv
    # analysis the PRD cites — treated there as a soft, single-preprint ceiling, not a hard
    # constant, hence the margin.
    context_token_budget: int = Field(default=1800)
    token_count_encoding: str = Field(default="cl100k_base")
    citation_snippet_max_chars: int = Field(default=280)

    # --- BYOK LLM (multi-provider via LiteLLM, per Q&A with user) ---
    default_llm_model: str = Field(default="openai/gpt-4o-mini")
    llm_request_timeout_s: float = Field(default=60.0)

    # --- Docling / parsing (Section 2 Stage 2, Section 8) ---
    docling_ocr_enabled: bool = Field(default=True)
    docling_picture_images_scale: float = Field(default=2.0)

    # --- Offline eval (Section 5.2) — a fixed judge, independent of any caller's BYOK key,
    # so scores are comparable run over run regardless of who is querying live. ---
    eval_judge_model: str = Field(default="openai/gpt-4o-mini")
    eval_judge_api_key: str | None = Field(default=None)


settings = Settings()
