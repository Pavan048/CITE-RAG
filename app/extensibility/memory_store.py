"""Seam for a future real memory module (PRD Section 0): "a later phase will introduce ... a real
memory module. Neither is built now, but every place conversational state would live MUST go
through the MemoryStore interface ... this is the only reason [it] exists."

Phase 1 has no multi-turn concept at all — `POST /v1/query` (Section 4) carries no session or
conversation id, and `RetrievalState` (retrieval/state.py) already holds everything one query
needs. So there is no real call site for this today, on purpose: forcing an unused dependency into
today's single-turn flow just to "use" the interface would be exactly the kind of half-finished,
speculative wiring the project's own conventions warn against. The interface is kept minimal and
ready to be picked up when multi-turn support is actually built.
"""

from __future__ import annotations

from typing import Protocol


class MemoryStore(Protocol):
    def get(self, session_key: str) -> list[dict[str, str]]:
        """Return prior turns for `session_key`, oldest first. Empty if none exist."""
        ...

    def append(self, session_key: str, role: str, content: str) -> None:
        """Record one turn for `session_key`."""
        ...


class NoOpMemoryStore:
    """Phase-1 implementation: satisfies the interface, retains nothing."""

    def get(self, session_key: str) -> list[dict[str, str]]:
        return []

    def append(self, session_key: str, role: str, content: str) -> None:
        return None


memory_store: MemoryStore = NoOpMemoryStore()
