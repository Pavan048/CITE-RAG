"""Thin client for Qdrant (PRD Section 1, Section 6, Section 7). Every raw Qdrant filter, payload
index, and collection config lives here — nothing in `retrieval/` or `ingestion/` builds a
`models.Filter` or talks to the SDK directly (code-quality rule 4).

RRF fusion (Section 3: "fuse with Reciprocal Rank Fusion, k=60 default") is implemented here in
Python rather than via Qdrant's server-side `FusionQuery`, because that server-side fusion doesn't
expose a configurable `k` — the PRD wants `k` tunable via config.py, so this runs two separate
named-vector searches (dense, sparse) and fuses their rank-ordered results by hand.

Point IDs for both collections must be valid Qdrant IDs (UUID string or unsigned int) — the
ingestion pipeline is responsible for minting `doc_id`/`chunk_id` as UUID4 strings so they double
as point IDs directly, with no separate id-mapping table needed.
"""

from __future__ import annotations

from qdrant_client import QdrantClient as _QdrantSDKClient
from qdrant_client import models
from pydantic import BaseModel

from app.config import settings
from app.models.mongo_models import ChunkPayload, DocumentSummaryPayload
from app.services.embedding_service import SparseVectorData

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class ChunkSearchResult(BaseModel):
    chunk_id: str
    payload: ChunkPayload
    fused_score: float


class DocumentSearchResult(BaseModel):
    doc_id: str
    score: float


class QdrantService:
    def __init__(self) -> None:
        self._client = _QdrantSDKClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def ensure_collections(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}

        if settings.chunks_collection not in existing:
            self._client.create_collection(
                collection_name=settings.chunks_collection,
                vectors_config={
                    _DENSE_VECTOR_NAME: models.VectorParams(size=settings.embedding_dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={_SPARSE_VECTOR_NAME: models.SparseVectorParams()},
            )
            for field in ("doc_id", "content_hash"):
                self._client.create_payload_index(
                    settings.chunks_collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                )

        if settings.documents_collection not in existing:
            self._client.create_collection(
                collection_name=settings.documents_collection,
                vectors_config={
                    _DENSE_VECTOR_NAME: models.VectorParams(size=settings.embedding_dim, distance=models.Distance.COSINE)
                },
            )
            for field in ("doc_id", "content_hash"):
                self._client.create_payload_index(
                    settings.documents_collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                )

    # --- Writes ---

    def upsert_chunks_batch(
        self, items: list[tuple[str, list[float], SparseVectorData, ChunkPayload]]
    ) -> None:
        if not items:
            return
        points = [
            models.PointStruct(
                id=chunk_id,
                vector={
                    _DENSE_VECTOR_NAME: dense,
                    _SPARSE_VECTOR_NAME: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=payload.model_dump(mode="json"),
            )
            for chunk_id, dense, sparse, payload in items
        ]
        self._client.upsert(collection_name=settings.chunks_collection, points=points)

    def upsert_document_summary(self, doc_id: str, dense: list[float], payload: DocumentSummaryPayload) -> None:
        self._client.upsert(
            collection_name=settings.documents_collection,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector={_DENSE_VECTOR_NAME: dense},
                    payload=payload.model_dump(mode="json"),
                )
            ],
        )

    # --- Reads ---

    def search_documents(self, query_dense: list[float], top_k: int, score_threshold: float) -> list[DocumentSearchResult]:
        result = self._client.query_points(
            collection_name=settings.documents_collection,
            query=query_dense,
            using=_DENSE_VECTOR_NAME,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=False,
        )
        return [DocumentSearchResult(doc_id=str(point.id), score=point.score) for point in result.points]

    def hybrid_search_chunks(
        self,
        query_dense: list[float],
        query_sparse: SparseVectorData,
        doc_ids: list[str],
        limit_per_list: int,
        rrf_k: int,
        fused_top_k: int,
    ) -> list[ChunkSearchResult]:
        query_filter = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])

        dense_points = self._client.query_points(
            collection_name=settings.chunks_collection,
            query=query_dense,
            using=_DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit_per_list,
            with_payload=True,
        ).points
        sparse_points = self._client.query_points(
            collection_name=settings.chunks_collection,
            query=models.SparseVector(indices=query_sparse.indices, values=query_sparse.values),
            using=_SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit_per_list,
            with_payload=True,
        ).points

        payload_by_id: dict[str, dict] = {}
        fused_scores: dict[str, float] = {}
        for point_list in (dense_points, sparse_points):
            for rank, point in enumerate(point_list, start=1):
                pid = str(point.id)
                payload_by_id.setdefault(pid, point.payload)
                fused_scores[pid] = fused_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)

        top = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:fused_top_k]
        return [
            ChunkSearchResult(chunk_id=pid, payload=ChunkPayload(**payload_by_id[pid]), fused_score=score)
            for pid, score in top
        ]

    # --- Deletes (Section 4: DELETE /v1/documents/{doc_id}) ---

    def delete_document(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=settings.chunks_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
        )
        self._client.delete(
            collection_name=settings.documents_collection,
            points_selector=models.PointIdsList(points=[doc_id]),
        )


qdrant_service = QdrantService()
