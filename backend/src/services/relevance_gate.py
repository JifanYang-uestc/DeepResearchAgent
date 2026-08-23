"""Backend-aware relevance gate for retrieved knowledge candidates."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from config import Configuration
from rag.types import RetrievalResult


@dataclass(frozen=True, slots=True)
class KnowledgeGateResult:
    """Accepted and rejected candidates plus an observable judgment reason."""

    accepted: list[RetrievalResult]
    rejected: list[RetrievalResult]
    reason: str
    mode: str
    threshold: float | None = None


class KnowledgeRelevanceGate:
    """Keep Top-K candidates out of evidence until relevance is established."""

    def __init__(self, config: Configuration) -> None:
        self._config = config

    def filter(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        backend_name: str,
    ) -> KnowledgeGateResult:
        """Filter candidates with backend-appropriate, non-fabricated semantics."""

        if not candidates:
            return KnowledgeGateResult(
                accepted=[],
                rejected=[],
                reason="Knowledge Retrieval 未返回候选证据。",
                mode="empty",
            )

        if backend_name == "helloagents":
            threshold = self._config.knowledge_relevance_threshold
            accepted = [item for item in candidates if item.score >= threshold]
            rejected = [item for item in candidates if item.score < threshold]
            reason = (
                f"Semantic cosine threshold={threshold:.2f}："
                f"接受 {len(accepted)}，拒绝 {len(rejected)}。"
            )
            return KnowledgeGateResult(
                accepted=_rerank(accepted),
                rejected=rejected,
                reason=reason,
                mode="semantic_threshold",
                threshold=threshold,
            )

        accepted = [item for item in candidates if _legacy_relevant(query, item)]
        rejected = [item for item in candidates if item not in accepted]
        return KnowledgeGateResult(
            accepted=_rerank(accepted),
            rejected=rejected,
            reason=(
                "Legacy backend 的 score 不与 Semantic cosine 比较；"
                f"文档名/词项判断接受 {len(accepted)}，拒绝 {len(rejected)}。"
            ),
            mode="legacy_lexical_judge",
        )


def _legacy_relevant(query: str, result: RetrievalResult) -> bool:
    query_terms = _terms(query)
    if not query_terms:
        return False
    document_terms = _terms(Path(result.chunk.document).stem)
    if query_terms & document_terms:
        return True
    content_terms = _terms(result.chunk.content)
    informative = {term for term in query_terms if len(term) >= 2}
    if not informative:
        return False
    overlap = informative & content_terms
    return len(overlap) >= max(1, min(2, len(informative)))


def _terms(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    latin = set(re.findall(r"[a-z0-9]+", normalized))
    cjk = "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized))
    bigrams = {cjk[index : index + 2] for index in range(max(0, len(cjk) - 1))}
    return latin | bigrams


def _rerank(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return [
        RetrievalResult(rank=rank, score=item.score, chunk=item.chunk)
        for rank, item in enumerate(results, start=1)
    ]
