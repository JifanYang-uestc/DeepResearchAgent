"""V0.2 HashingEmbedding + FAISS backend retained for compatibility."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from config import Configuration

from .base import KnowledgeDocumentInfo
from .catalog import build_knowledge_catalog
from .chunker import chunk_documents
from .loader import load_documents
from .retriever import KnowledgeRetriever
from .types import RetrievalResult
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]


class LegacyFaissBackend:
    """Lazy lifecycle adapter for the stable V0.2 retrieval implementation."""

    name = "legacy_faiss"

    def __init__(self, config: Configuration) -> None:
        self._config = config
        self._store: FaissVectorStore | None = None
        self._retriever: KnowledgeRetriever | None = None
        self._lock = Lock()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve V0.2 lexical-hashing candidates."""

        return self._get_retriever().retrieve(query, top_k=top_k)

    def health_check(self) -> bool:
        """Return whether the index can be loaded or built."""

        try:
            self._get_retriever()
        except Exception as exc:  # noqa: BLE001 - health boundary
            logger.warning("Legacy FAISS health check failed: %s", exc)
            return False
        return True

    def get_catalog(self) -> list[KnowledgeDocumentInfo]:
        """Build a compact catalog without loading the vector index."""

        return build_knowledge_catalog(
            resolve_backend_path(self._config.knowledge_base_path)
        )

    def _get_retriever(self) -> KnowledgeRetriever:
        if self._retriever is not None:
            return self._retriever

        with self._lock:
            if self._retriever is not None:
                return self._retriever

            index_path = resolve_backend_path(self._config.knowledge_index_path)
            store = FaissVectorStore()
            try:
                store.load(index_path)
            except (FileNotFoundError, ValueError):
                if not self._config.knowledge_auto_build:
                    raise
                knowledge_path = resolve_backend_path(self._config.knowledge_base_path)
                pages = load_documents(knowledge_path)
                chunks = chunk_documents(
                    pages,
                    chunk_size=self._config.knowledge_chunk_size,
                    chunk_overlap=self._config.knowledge_chunk_overlap,
                )
                store.build(chunks)
                store.save(index_path)
                logger.info(
                    "Built legacy knowledge index: pages=%s chunks=%s path=%s",
                    len(pages),
                    len(chunks),
                    index_path,
                )

            self._store = store
            self._retriever = KnowledgeRetriever(
                store,
                top_k=self._config.knowledge_top_k,
                minimum_score=self._config.knowledge_minimum_score,
            )
            return self._retriever


def resolve_backend_path(value: str) -> Path:
    """Resolve configuration paths consistently from common launch directories."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    working_path = (Path.cwd() / path).resolve()
    if working_path.exists():
        return working_path
    return (BACKEND_DIR / path).resolve()
