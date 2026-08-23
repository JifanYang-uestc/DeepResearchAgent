"""Knowledge retriever built on the persisted FAISS vector store."""

from __future__ import annotations

from pathlib import Path

from .embedding import HashingEmbedding
from .types import RetrievalResult
from .vector_store import FaissVectorStore


class KnowledgeRetriever:
    """Retrieve the most relevant local knowledge chunks for a query."""

    def __init__(
        self,
        store: FaissVectorStore,
        *,
        top_k: int = 5,
        minimum_score: float = 0.0,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.store = store
        self.top_k = top_k
        self.minimum_score = minimum_score

    @classmethod
    def from_index(
        cls,
        index_directory: str | Path,
        *,
        top_k: int = 5,
        minimum_score: float = 0.0,
        embedding: HashingEmbedding | None = None,
    ) -> KnowledgeRetriever:
        """Load a persisted index and construct a retriever."""

        store = FaissVectorStore(embedding)
        store.load(index_directory)
        return cls(store, top_k=top_k, minimum_score=minimum_score)

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        """Search local knowledge and apply the configured score threshold."""

        limit = self.top_k if top_k is None else top_k
        results = self.store.search(query, top_k=limit)
        return [result for result in results if result.score >= self.minimum_score]
