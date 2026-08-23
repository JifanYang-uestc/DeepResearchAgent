"""Typed records shared by the local RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text extracted from one logical source page."""

    content: str
    document: str
    source_path: str
    file_type: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A retrievable text chunk with traceable source metadata."""

    chunk_id: str
    content: str
    document: str
    source_path: str
    file_type: str
    page: int | None
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KnowledgeChunk:
        """Restore a chunk saved in index metadata."""

        return cls(**payload)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked chunk returned by vector search."""

    rank: int
    score: float
    chunk: KnowledgeChunk
