"""Dual-route evidence collection for Knowledge RAG and Web Search."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal

from config import Configuration
from rag.types import RetrievalResult
from services.knowledge import KnowledgeService
from services.relevance_gate import KnowledgeGateResult, KnowledgeRelevanceGate
from services.retrieval_router import RetrievalRoute, RoutingDecision
from services.search import (
    dispatch_search,
    normalize_search_response,
    prepare_research_context,
)
from services.user_messages import WEB_PROVIDER_NOTICE, WEB_UNAVAILABLE

logger = logging.getLogger(__name__)

WebSearch = Callable[
    [str, Configuration, int],
    tuple[dict[str, Any] | None, list[str], str | None, str],
]


@dataclass(slots=True)
class EvidenceSource:
    """A structured, frontend-safe citation for one evidence item."""

    type: Literal["knowledge", "web"]
    title: str
    snippet: str = ""
    url: str = ""
    document: str = ""
    page: int | None = None
    chunk_id: str = ""
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe citation payload."""

        return {
            "type": self.type,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "document": self.document,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


@dataclass(slots=True)
class EvidenceBundle:
    """Combined local and web evidence for one research TODO."""

    context: str = ""
    sources_summary: str = ""
    notices: list[str] = field(default_factory=list)
    backend: str = "none"
    knowledge_results: list[RetrievalResult] = field(default_factory=list)
    web_result: dict[str, Any] | None = None
    sources: list[EvidenceSource] = field(default_factory=list)
    routing_decision: RoutingDecision | None = None
    gate_result: KnowledgeGateResult | None = None
    timings_ms: dict[str, float | None] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Return whether at least one evidence route produced usable context."""

        return bool(self.context.strip())


def gather_research_evidence(
    query: str,
    config: Configuration,
    loop_count: int,
    knowledge: KnowledgeService,
    *,
    web_search: WebSearch = dispatch_search,
    decision: RoutingDecision | None = None,
    relevance_gate: KnowledgeRelevanceGate | None = None,
) -> EvidenceBundle:
    """Execute the selected routes and admit only relevant knowledge evidence."""

    retrieval_started = perf_counter()
    decision = decision or RoutingDecision(
        route=RetrievalRoute.HYBRID,
        reason="Compatibility default: execute both V0.2 evidence routes.",
        confidence=1.0,
        knowledge_query=query,
        web_query=query,
        freshness_required=False,
    )
    relevance_gate = relevance_gate or KnowledgeRelevanceGate(config)
    notices: list[str] = []
    knowledge_results: list[RetrievalResult] = []
    gate_result: KnowledgeGateResult | None = None
    knowledge_backend = "none"
    knowledge_latency_ms: float | None = None
    web_latency_ms: float | None = None

    run_knowledge = decision.route in (
        RetrievalRoute.KNOWLEDGE,
        RetrievalRoute.HYBRID,
    )
    run_web = decision.route in (RetrievalRoute.WEB, RetrievalRoute.HYBRID)

    if run_knowledge:
        knowledge_started = perf_counter()
        knowledge_query = decision.knowledge_query or query
        knowledge_results, knowledge_notices, knowledge_backend = _retrieve_candidates(
            knowledge,
            knowledge_query,
            config.knowledge_probe_top_k,
        )
        notices.extend(knowledge_notices)
        gate_result = relevance_gate.filter(
            knowledge_query,
            knowledge_results,
            backend_name=knowledge_backend,
        )
        knowledge_results = gate_result.accepted
        if gate_result.rejected:
            notices.append(f"Knowledge Evidence 已拒绝：{gate_result.reason}")
        if decision.route is RetrievalRoute.KNOWLEDGE and not knowledge_results:
            run_web = True
            notices.append(
                "Knowledge 路由未获得有效 Evidence，已回退到 Web Search。"
            )
        knowledge_latency_ms = (perf_counter() - knowledge_started) * 1000

    knowledge_sources, knowledge_context, structured_knowledge = _format_knowledge(
        knowledge_results
    )

    web_result: dict[str, Any] | None = None
    web_sources = ""
    web_context = ""
    web_backend = "web"
    structured_web: list[EvidenceSource] = []
    if run_web:
        web_started = perf_counter()
        try:
            web_result, web_notices, answer_text, web_backend = web_search(
                decision.web_query or query,
                config,
                loop_count,
            )
            notices.extend(web_notices)
            web_result, response_issue = normalize_search_response(
                web_result,
                web_backend,
            )
            if response_issue and WEB_PROVIDER_NOTICE not in notices:
                notices.append(WEB_PROVIDER_NOTICE)
            answer_text = web_result.get("answer")
            web_backend = str(web_result.get("backend") or web_backend)
            if web_result and web_result.get("results"):
                web_sources, web_context = prepare_research_context(
                    web_result,
                    answer_text,
                    config,
                )
                structured_web = _web_sources(web_result)
        except Exception:  # noqa: BLE001 - degradation boundary
            logger.exception("Web Search failed during evidence collection")
            notices.append(WEB_UNAVAILABLE)
        finally:
            web_latency_ms = (perf_counter() - web_started) * 1000

    contexts = [part for part in (knowledge_context, web_context) if part]
    sources = [part for part in (knowledge_sources, web_sources) if part]
    routes = []
    if knowledge_context:
        routes.append("knowledge")
    if web_context:
        routes.append(web_backend)

    return EvidenceBundle(
        context="\n\n".join(contexts),
        sources_summary="\n".join(sources),
        notices=notices,
        backend="+".join(routes) if routes else "none",
        knowledge_results=knowledge_results,
        web_result=web_result,
        sources=[*structured_knowledge, *structured_web],
        routing_decision=decision,
        gate_result=gate_result,
        timings_ms={
            "knowledge_retrieval": knowledge_latency_ms,
            "web_search": web_latency_ms,
            "total_retrieval": (perf_counter() - retrieval_started) * 1000,
        },
    )


def _retrieve_candidates(
    knowledge: KnowledgeService,
    query: str,
    top_k: int,
) -> tuple[list[RetrievalResult], list[str], str]:
    retrieve_with_backend = getattr(knowledge, "retrieve_with_backend", None)
    if callable(retrieve_with_backend):
        return retrieve_with_backend(query, top_k=top_k)
    results, notices = knowledge.retrieve(query)
    return results[:top_k], notices, "helloagents"


def _format_knowledge(
    results: list[RetrievalResult],
) -> tuple[str, str, list[EvidenceSource]]:
    source_lines: list[str] = []
    contexts: list[str] = []
    sources: list[EvidenceSource] = []
    for result in results:
        chunk = result.chunk
        location = f"Page {chunk.page}" if chunk.page is not None else "Page N/A"
        source_lines.append(
            f"[Knowledge] {chunk.document} ({location}, Chunk {chunk.chunk_id}, "
            f"Score {result.score:.4f})"
        )
        contexts.append(
            f"[Knowledge Evidence {result.rank}]\n"
            f"Document: {chunk.document}\n"
            f"Page: {chunk.page if chunk.page is not None else 'N/A'}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Score: {result.score:.6f}\n"
            f"Content:\n{chunk.content}"
        )
        sources.append(
            EvidenceSource(
                type="knowledge",
                title=chunk.document,
                snippet=chunk.content,
                document=chunk.document,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                score=result.score,
            )
        )
    return "\n".join(source_lines), "\n\n".join(contexts), sources


def _web_sources(search_result: dict[str, Any]) -> list[EvidenceSource]:
    return [
        EvidenceSource(
            type="web",
            title=str(item.get("title") or item.get("url") or "Web source"),
            url=str(item.get("url") or ""),
            snippet=str(item.get("content") or item.get("raw_content") or ""),
        )
        for item in search_result.get("results", [])
        if item.get("url")
    ]
