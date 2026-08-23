"""Lifecycle service for the local knowledge index and retriever."""

from __future__ import annotations

import logging

from config import Configuration
from rag.base import KnowledgeBackend, KnowledgeBuildResult, KnowledgeDocumentInfo
from rag.helloagents_backend import HelloAgentsSemanticBackend
from rag.legacy_faiss_backend import LegacyFaissBackend
from rag.types import RetrievalResult
from services.log_redaction import redact_sensitive_text
from services.user_messages import (
    CATALOG_FALLBACK,
    CATALOG_UNAVAILABLE,
    KNOWLEDGE_FALLBACK,
    KNOWLEDGE_UNAVAILABLE,
)

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Load/build the local index once and degrade safely on failure."""

    def __init__(
        self,
        config: Configuration,
        backend: KnowledgeBackend | None = None,
        fallback_backend: KnowledgeBackend | None = None,
    ) -> None:
        self._config = config
        if backend is not None:
            self._backend = backend
            self._fallback_backend = fallback_backend
        elif config.knowledge_backend == "helloagents":
            self._backend = HelloAgentsSemanticBackend(config)
            self._fallback_backend = fallback_backend
        elif config.knowledge_backend == "legacy_faiss":
            self._backend = LegacyFaissBackend(config)
            self._fallback_backend = fallback_backend
        else:
            raise ValueError(
                "KNOWLEDGE_BACKEND must be 'helloagents' or 'legacy_faiss'"
            )

    def retrieve(self, query: str) -> tuple[list[RetrievalResult], list[str]]:
        """Return local evidence plus non-fatal availability notices."""

        results, notices, _ = self.retrieve_with_backend(query)
        return results, notices

    def retrieve_with_backend(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[list[RetrievalResult], list[str], str]:
        """Return candidates, notices, and the backend that produced them."""

        if not self._config.enable_knowledge_rag:
            return [], ["Knowledge RAG 已禁用，继续使用 Web Search。"], "disabled"

        limit = top_k or self._config.knowledge_top_k
        try:
            return self._backend.retrieve(query, limit), [], self._backend.name
        except Exception as exc:  # noqa: BLE001 - degradation boundary
            logger.error(
                "Knowledge backend %s unavailable: %s",
                self._backend.name,
                redact_sensitive_text(exc),
            )
            if self._fallback_backend is not None:
                try:
                    results = self._fallback_backend.retrieve(
                        query, limit
                    )
                    return results, [KNOWLEDGE_FALLBACK], self._fallback_backend.name
                except Exception as exc:  # noqa: BLE001 - fallback boundary
                    logger.error(
                        "Fallback knowledge backend %s unavailable: %s",
                        self._fallback_backend.name,
                        redact_sensitive_text(exc),
                    )
            return [], [KNOWLEDGE_UNAVAILABLE], "none"

    def get_catalog(self) -> tuple[list[KnowledgeDocumentInfo], list[str]]:
        """Return catalog metadata without exposing backend details upstream."""

        if not self._config.enable_knowledge_rag:
            return [], ["Knowledge RAG 已禁用，Knowledge Catalog 不可用。"]
        try:
            return self._backend.get_catalog(), []
        except Exception as exc:  # noqa: BLE001 - degradation boundary
            logger.error(
                "Knowledge catalog unavailable: %s",
                redact_sensitive_text(exc),
            )
            if self._fallback_backend is not None:
                try:
                    catalog = self._fallback_backend.get_catalog()
                    return catalog, [CATALOG_FALLBACK]
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Fallback knowledge catalog unavailable: %s",
                        redact_sensitive_text(exc),
                    )
            return [], [CATALOG_UNAVAILABLE]

    def prepare(self) -> tuple[bool, list[str]]:
        """Load or build the configured primary backend for setup commands."""

        if not self._config.enable_knowledge_rag:
            return False, ["Knowledge RAG 已禁用，无法准备索引。"]
        if self._backend.health_check():
            return True, []
        return False, [f"Knowledge Backend {self._backend.name} 准备失败，请检查日志。"]

    def rebuild(self) -> KnowledgeBuildResult:
        """Explicitly replace the selected index from the current corpus."""

        if not self._config.enable_knowledge_rag:
            raise RuntimeError("Knowledge RAG is disabled; cannot rebuild the index")
        return self._backend.rebuild()

    @property
    def backend_name(self) -> str:
        """Expose the configured backend name for diagnostics."""

        return self._backend.name
