"""Lifecycle service for the local knowledge index and retriever."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from config import Configuration
from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.retriever import KnowledgeRetriever
from rag.types import RetrievalResult
from rag.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]


class KnowledgeService:
    """Load/build the local index once and degrade safely on failure."""

    def __init__(self, config: Configuration) -> None:
        self._config = config
        self._retriever: KnowledgeRetriever | None = None
        self._lock = Lock()

    def retrieve(self, query: str) -> tuple[list[RetrievalResult], list[str]]:
        """Return local evidence plus non-fatal availability notices."""

        if not self._config.enable_knowledge_rag:
            return [], ["Knowledge RAG 已禁用，继续使用 Web Search。"]

        try:
            retriever = self._get_retriever()
            return retriever.retrieve(query), []
        except Exception as exc:  # noqa: BLE001 - degradation boundary
            logger.warning("Knowledge RAG unavailable; degrading to web search: %s", exc)
            return [], [f"Knowledge RAG 不可用，已退化到 Web Search：{exc}"]

    def _get_retriever(self) -> KnowledgeRetriever:
        if self._retriever is not None:
            return self._retriever

        with self._lock:
            if self._retriever is not None:
                return self._retriever

            index_path = _resolve_backend_path(self._config.knowledge_index_path)
            store = FaissVectorStore()
            try:
                store.load(index_path)
            except FileNotFoundError:
                if not self._config.knowledge_auto_build:
                    raise
                knowledge_path = _resolve_backend_path(self._config.knowledge_base_path)
                pages = load_documents(knowledge_path)
                chunks = chunk_documents(
                    pages,
                    chunk_size=self._config.knowledge_chunk_size,
                    chunk_overlap=self._config.knowledge_chunk_overlap,
                )
                store.build(chunks)
                store.save(index_path)
                logger.info(
                    "Built local knowledge index: pages=%s chunks=%s path=%s",
                    len(pages),
                    len(chunks),
                    index_path,
                )

            self._retriever = KnowledgeRetriever(
                store,
                top_k=self._config.knowledge_top_k,
                minimum_score=self._config.knowledge_minimum_score,
            )
            return self._retriever


def _resolve_backend_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    working_path = (Path.cwd() / path).resolve()
    if working_path.exists():
        return working_path
    return (BACKEND_DIR / path).resolve()
