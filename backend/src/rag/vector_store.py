"""FAISS vector storage and persistence for local knowledge chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import faiss

from .types import KnowledgeChunk, RetrievalResult

INDEX_FILENAME = "knowledge.faiss"
METADATA_FILENAME = "metadata.json"
INDEX_VERSION = 2


class VectorEmbedding(Protocol):
    """Embedding contract required by the FAISS persistence layer."""

    dimensions: int

    @property
    def fingerprint(self) -> str:
        """Return a stable model identity."""

    def embed_many(self, texts: Any) -> Any:
        """Embed a batch as a FAISS-compatible matrix."""


class FaissVectorStore:
    """A persisted cosine-similarity FAISS index plus chunk metadata."""

    def __init__(self, embedding: VectorEmbedding | None = None) -> None:
        if embedding is None:
            from .embedding import HashingEmbedding

            embedding = HashingEmbedding()
        self.embedding = embedding
        self._index: Any | None = None
        self._chunks: list[KnowledgeChunk] = []

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._chunks)

    @property
    def chunks(self) -> tuple[KnowledgeChunk, ...]:
        """Expose immutable chunk metadata for catalog construction."""

        return tuple(self._chunks)

    def build(self, chunks: list[KnowledgeChunk]) -> None:
        """Build a fresh inner-product index from normalized embeddings."""

        if not chunks:
            raise ValueError("Cannot build a knowledge index without chunks")
        vectors = self.embedding.embed_many(_embedding_text(chunk) for chunk in chunks)
        index = faiss.IndexFlatIP(self.embedding.dimensions)
        index.add(vectors)
        self._index = index
        self._chunks = list(chunks)

    def save(self, directory: str | Path) -> None:
        """Persist FAISS data and traceable chunk metadata."""

        if self._index is None or not self._chunks:
            raise RuntimeError("Build or load an index before saving it")

        target = Path(directory).resolve()
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target / INDEX_FILENAME))
        payload = {
            "version": INDEX_VERSION,
            "embedding": self.embedding.fingerprint,
            "dimensions": self.embedding.dimensions,
            "count": len(self._chunks),
            "chunks": [chunk.to_dict() for chunk in self._chunks],
        }
        metadata_path = target / METADATA_FILENAME
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)

    def load(self, directory: str | Path) -> None:
        """Load an existing index and reject incompatible metadata."""

        source = Path(directory).resolve()
        index_path = source / INDEX_FILENAME
        metadata_path = source / METADATA_FILENAME
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Knowledge index is incomplete: {source}")

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION:
            raise ValueError("Knowledge index version is incompatible; rebuild it")
        if payload.get("embedding") != self.embedding.fingerprint:
            raise ValueError("Knowledge index was created with a different embedding")
        chunks = [KnowledgeChunk.from_dict(item) for item in payload.get("chunks", [])]
        index = faiss.read_index(str(index_path))
        if index.d != self.embedding.dimensions or index.ntotal != len(chunks):
            raise ValueError("Knowledge index and metadata counts are inconsistent")
        self._index = index
        self._chunks = chunks

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return top-k chunks ranked by cosine similarity."""

        if self._index is None:
            raise RuntimeError("Build or load an index before searching")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self._chunks:
            return []

        query_vector = self.embedding.embed_many([query])
        scores, positions = self._index.search(query_vector, min(top_k, len(self._chunks)))
        results: list[RetrievalResult] = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position < 0:
                continue
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(score),
                    chunk=self._chunks[int(position)],
                )
            )
        return results


def _embedding_text(chunk: KnowledgeChunk) -> str:
    """Add source identity so method-specific TODOs prefer their primary paper."""

    document_identity = Path(chunk.document).stem.replace("_", " ").replace("-", " ")
    return f"{document_identity} {document_identity} {document_identity}\n{chunk.content}"
