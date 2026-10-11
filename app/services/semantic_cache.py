"""In-process semantic cache for the retrieval pipeline.

Caches complete query results (answer + citations + context) keyed by semantic similarity of
the query embedding rather than exact string match — "what is the refund policy?" and "tell me
about your refund policy" should hit the same cache entry.

Design choices:
- **No Redis dependency**: Uses an in-process LRU dict with a fixed capacity. Simple, zero
  infrastructure, correct for a single-process deployment (FastAPI + uvicorn). A Redis-backed
  variant would be needed for multi-worker deployments where cache coherence across workers matters
  — that's a separate concern and the interface here is designed to make swapping trivial.
- **Cosine similarity threshold**: Two queries are considered equivalent if their dense embedding
  vectors have cosine similarity >= `semantic_cache_similarity_threshold` (default 0.97). This is
  intentionally conservative: 0.97 means near-identical phrasing, not just related topics. Lower
  values hit more often but risk serving the wrong cached answer to a meaningfully different query.
- **Scope isolation**: Cache key includes the `file_ids` scope. Query A scoped to doc-1 and the
  same query scoped to doc-2 must not share a cache entry — different corpora, potentially
  different answers.
- **TTL via insertion timestamp**: Each entry stores `inserted_at`. `get()` silently evicts
  entries older than `semantic_cache_ttl_seconds` rather than using a background expiry thread,
  which would complicate asyncio lifetime management.
- **Cache miss on embedding failure**: If the embedding service is unreachable, `get()` returns
  None and `put()` is a no-op — the cache degrades gracefully rather than blocking the query.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.models.schemas import Citation

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query_embedding: list[float]
    answer: str
    citations: list[Citation]
    partial_answer: bool
    routed_docs: list[str] | None
    inserted_at: float = field(default_factory=time.monotonic)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _scope_key(file_ids: list[str] | None) -> str:
    """Stable string representation of the query scope for cache partitioning."""
    if not file_ids:
        return "__all__"
    return ",".join(sorted(file_ids))


class SemanticCache:
    """Thread-safe (GIL-protected dict ops) in-process semantic cache.

    Internally an OrderedDict used as an LRU: on every hit the entry is moved to the end
    (most recently used), and when capacity is exceeded the first item (least recently used)
    is evicted.
    """

    def __init__(
        self,
        capacity: int | None = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._capacity = capacity or settings.semantic_cache_max_entries
        self._threshold = similarity_threshold or settings.semantic_cache_similarity_threshold
        self._ttl = ttl_seconds or settings.semantic_cache_ttl_seconds
        # scope_key -> OrderedDict[entry_index, CacheEntry]; outer dict partitions by scope,
        # inner LRU dict holds entries for that scope.
        self._store: dict[str, OrderedDict[int, CacheEntry]] = {}
        self._counter = 0  # monotonic entry ID, used as LRU dict key

    def get(
        self,
        query_embedding: list[float],
        file_ids: list[str] | None,
    ) -> CacheEntry | None:
        """Returns a cached entry if a semantically equivalent query exists within TTL.

        Scans all entries for the given scope, evicts expired ones, and returns the first
        entry whose cosine similarity to `query_embedding` meets the threshold.
        Returns None on cache miss, expired scope, or empty scope.
        """
        if not settings.semantic_cache_enabled:
            return None

        scope = _scope_key(file_ids)
        bucket = self._store.get(scope)
        if not bucket:
            return None

        now = time.monotonic()
        expired_keys = [k for k, e in bucket.items() if (now - e.inserted_at) > self._ttl]
        for k in expired_keys:
            del bucket[k]

        for key, entry in list(bucket.items()):
            sim = _cosine_similarity(query_embedding, entry.query_embedding)
            if sim >= self._threshold:
                # Move to end of LRU
                bucket.move_to_end(key)
                logger.debug("semantic_cache hit (sim=%.4f scope=%s)", sim, scope)
                return entry

        return None

    def put(
        self,
        query_embedding: list[float],
        file_ids: list[str] | None,
        answer: str,
        citations: list[Citation],
        partial_answer: bool,
        routed_docs: list[str] | None,
    ) -> None:
        """Stores a query result. Evicts the LRU entry if capacity is exceeded."""
        if not settings.semantic_cache_enabled:
            return

        scope = _scope_key(file_ids)
        if scope not in self._store:
            self._store[scope] = OrderedDict()

        bucket = self._store[scope]
        self._counter += 1
        bucket[self._counter] = CacheEntry(
            query_embedding=query_embedding,
            answer=answer,
            citations=citations,
            partial_answer=partial_answer,
            routed_docs=routed_docs,
        )
        # Evict LRU if over capacity (across all scopes combined)
        total = sum(len(b) for b in self._store.values())
        if total > self._capacity:
            # Find and evict the globally oldest entry
            oldest_scope = min(self._store, key=lambda s: next(iter(self._store[s])))
            oldest_bucket = self._store[oldest_scope]
            oldest_bucket.popitem(last=False)
            if not oldest_bucket:
                del self._store[oldest_scope]
        logger.debug("semantic_cache stored (scope=%s total=%d)", scope, total)

    def invalidate_scope(self, file_ids: list[str] | None) -> None:
        """Evicts all entries for a given document scope.

        Should be called after a document is deleted so cached answers referencing that document
        are not served from the cache after its chunks are gone from Qdrant.
        """
        scope = _scope_key(file_ids)
        self._store.pop(scope, None)

    def clear(self) -> None:
        self._store.clear()
        self._counter = 0


semantic_cache = SemanticCache()
