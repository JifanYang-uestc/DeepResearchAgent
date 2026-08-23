"""Lifecycle service for the local knowledge index and retriever."""

from __future__ import annotations

import logging

from config import Configuration
from rag.base import KnowledgeBackend, KnowledgeDocumentInfo
from rag.helloagents_backend import HelloAgentsSemanticBackend
from rag.legacy_faiss_backend import LegacyFaissBackend
from rag.types import RetrievalResult

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
            self._fallback_backend = fallback_backend or LegacyFaissBackend(config)
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
            logger.warning(
                "Knowledge backend %s unavailable; degrading to web search: %s",
                self._backend.name,
                exc,
            )
            if self._fallback_backend is not None:
                try:
                    results = self._fallback_backend.retrieve(
                        query, limit
                    )
                    notice = (
                        f"Knowledge Backend {self._backend.name} 不可用，"
                        f"已回退到 {self._fallback_backend.name}：{exc}"
                    )
                    return results, [notice], self._fallback_backend.name
                except Exception as fallback_exc:  # noqa: BLE001 - fallback boundary
                    logger.warning(
                        "Fallback knowledge backend %s unavailable: %s",
                        self._fallback_backend.name,
                        fallback_exc,
                    )
            return [], [f"Knowledge RAG 不可用，已退化到 Web Search：{exc}"], "none"

    def get_catalog(self) -> tuple[list[KnowledgeDocumentInfo], list[str]]:
        """Return catalog metadata without exposing backend details upstream."""

        if not self._config.enable_knowledge_rag:
            return [], ["Knowledge RAG 已禁用，Knowledge Catalog 不可用。"]
        try:
            return self._backend.get_catalog(), []
        except Exception as exc:  # noqa: BLE001 - degradation boundary
            logger.warning("Knowledge catalog unavailable: %s", exc)
            if self._fallback_backend is not None:
                try:
                    catalog = self._fallback_backend.get_catalog()
                    return catalog, [
                        f"Knowledge Catalog 已回退到 {self._fallback_backend.name}：{exc}"
                    ]
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.warning("Fallback knowledge catalog unavailable: %s", fallback_exc)
            return [], [f"Knowledge Catalog 不可用：{exc}"]

    @property
    def backend_name(self) -> str:
        """Expose the configured backend name for diagnostics."""

        return self._backend.name
