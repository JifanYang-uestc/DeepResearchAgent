"""Semantic knowledge backend powered by HelloAgents local embeddings."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Protocol

import numpy as np

from config import Configuration

from .base import KnowledgeBuildResult, KnowledgeDocumentInfo
from .catalog import build_knowledge_catalog
from .chunker import chunk_documents
from .legacy_faiss_backend import resolve_backend_path
from .loader import load_documents
from .retriever import KnowledgeRetriever
from .types import RetrievalResult
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)
SEMANTIC_INDEX_DIRECTORY = "helloagents-semantic"


class SemanticEmbedding(Protocol):
    """Embedding surface required by the persisted FAISS store."""

    dimensions: int

    @property
    def fingerprint(self) -> str:
        """Return a stable persistence identifier."""

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Return normalized float32 rows."""


class HelloAgentsLocalEmbedding:
    """Normalize HelloAgents LocalTransformerEmbedding for cosine FAISS."""

    def __init__(self, model_name: str) -> None:
        from hello_agents.memory.embedding import LocalTransformerEmbedding

        self.model_name = model_name
        self._embedding = LocalTransformerEmbedding(model_name=model_name)
        self.dimensions = self._embedding.dimension
        if self.dimensions <= 0:
            raise RuntimeError("HelloAgents embedding returned an invalid dimension")

    @property
    def fingerprint(self) -> str:
        """Identify the exact semantic model used to build an index."""

        return f"helloagents-local-transformer:{self.model_name}:{self.dimensions}"

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Encode a batch and normalize vectors for cosine similarity."""

        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors = np.asarray(self._embedding.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Semantic embedding dimension changed: "
                f"expected {self.dimensions}, received {vectors.shape[1]}"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Semantic embedding produced a zero vector")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class HelloAgentsSemanticBackend:
    """Page-aware semantic retrieval using HelloAgents embedding and FAISS."""

    name = "helloagents"

    def __init__(
        self,
        config: Configuration,
        *,
        embedding: SemanticEmbedding | None = None,
    ) -> None:
        self._config = config
        self._embedding = embedding
        self._store: FaissVectorStore | None = None
        self._retriever: KnowledgeRetriever | None = None
        self._lock = Lock()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return semantic candidates while preserving project metadata."""

        return self._get_retriever().retrieve(query, top_k=top_k)

    def health_check(self) -> bool:
        """Return whether the semantic model and persisted index are available."""

        try:
            self._get_retriever()
        except Exception as exc:  # noqa: BLE001 - health boundary
            logger.warning("HelloAgents semantic backend health check failed: %s", exc)
            return False
        return True

    def get_catalog(self) -> list[KnowledgeDocumentInfo]:
        """Return document metadata without loading the semantic model."""

        return build_knowledge_catalog(
            resolve_backend_path(self._config.knowledge_base_path)
        )

    def rebuild(self) -> KnowledgeBuildResult:
        """Replace the semantic index from the current knowledge corpus."""

        with self._lock:
            if self._config.embedding_provider != "local_transformer":
                raise ValueError(
                    "V0.3 helloagents backend supports "
                    "EMBEDDING_PROVIDER=local_transformer"
                )
            embedding = self._embedding or HelloAgentsLocalEmbedding(
                self._config.embedding_model
            )
            store, result = self._build_store(embedding)
            self._embedding = embedding
            self._store = store
            self._retriever = self._create_retriever(store)
            return result

    def _get_retriever(self) -> KnowledgeRetriever:
        if self._retriever is not None:
            return self._retriever

        with self._lock:
            if self._retriever is not None:
                return self._retriever
            if self._config.embedding_provider != "local_transformer":
                raise ValueError(
                    "V0.3 helloagents backend supports EMBEDDING_PROVIDER=local_transformer"
                )

            embedding = self._embedding or HelloAgentsLocalEmbedding(
                self._config.embedding_model
            )
            index_root = resolve_backend_path(self._config.knowledge_index_path)
            index_path = index_root / SEMANTIC_INDEX_DIRECTORY
            store = FaissVectorStore(embedding)
            try:
                store.load(index_path)
            except (FileNotFoundError, ValueError):
                if not self._config.knowledge_auto_build:
                    raise
                store, _ = self._build_store(embedding)

            self._embedding = embedding
            self._store = store
            self._retriever = self._create_retriever(store)
            return self._retriever

    def _build_store(
        self,
        embedding: SemanticEmbedding,
    ) -> tuple[FaissVectorStore, KnowledgeBuildResult]:
        knowledge_path = resolve_backend_path(self._config.knowledge_base_path)
        index_root = resolve_backend_path(self._config.knowledge_index_path)
        index_path = index_root / SEMANTIC_INDEX_DIRECTORY
        pages = load_documents(knowledge_path)
        chunks = chunk_documents(
            pages,
            chunk_size=self._config.knowledge_chunk_size,
            chunk_overlap=self._config.knowledge_chunk_overlap,
        )
        store = FaissVectorStore(embedding)
        store.build(chunks)
        store.save(index_path)
        result = KnowledgeBuildResult(
            backend=self.name,
            document_count=len({page.source_path for page in pages}),
            page_count=len(pages),
            chunk_count=len(chunks),
            index_path=str(index_path),
        )
        logger.info(
            "Built semantic knowledge index: model=%s documents=%s pages=%s "
            "chunks=%s path=%s",
            self._config.embedding_model,
            result.document_count,
            result.page_count,
            result.chunk_count,
            result.index_path,
        )
        return store, result

    def _create_retriever(self, store: FaissVectorStore) -> KnowledgeRetriever:
        return KnowledgeRetriever(
                store,
                top_k=self._config.knowledge_top_k,
                minimum_score=self._config.knowledge_minimum_score,
            )
