"""Stable interfaces shared by local knowledge retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .types import RetrievalResult


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentInfo:
    """Truthful, compact metadata exposed to the retrieval router."""

    document: str
    file_type: str
    pages: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe catalog entry."""

        return {
            "document": self.document,
            "type": self.file_type,
            "pages": self.pages,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeBuildResult:
    """Typed summary returned by an explicit corpus rebuild."""

    backend: str
    document_count: int
    page_count: int
    chunk_count: int
    index_path: str


@runtime_checkable
class KnowledgeBackend(Protocol):
    """Backend contract kept behind :class:`KnowledgeService`."""

    name: str

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return ranked knowledge candidates for a query."""

    def health_check(self) -> bool:
        """Return whether the backend can currently serve retrieval."""

    def get_catalog(self) -> list[KnowledgeDocumentInfo]:
        """Return a chunk-free summary of indexed documents."""

    def rebuild(self) -> KnowledgeBuildResult:
        """Replace the persisted index from the current knowledge corpus."""
