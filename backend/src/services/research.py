"""Dual-route evidence collection for Knowledge RAG and Web Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from config import Configuration
from rag.types import RetrievalResult
from services.knowledge import KnowledgeService
from services.search import dispatch_search, prepare_research_context

WebSearch = Callable[
    [str, Configuration, int],
    tuple[dict[str, Any] | None, list[str], str | None, str],
]


@dataclass(slots=True)
class EvidenceBundle:
    """Combined local and web evidence for one research TODO."""

    context: str = ""
    sources_summary: str = ""
    notices: list[str] = field(default_factory=list)
    backend: str = "none"
    knowledge_results: list[RetrievalResult] = field(default_factory=list)
    web_result: dict[str, Any] | None = None

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
) -> EvidenceBundle:
    """Collect both routes independently so either one can keep research running."""

    knowledge_results, notices = knowledge.retrieve(query)
    knowledge_sources, knowledge_context = _format_knowledge(knowledge_results)

    web_result: dict[str, Any] | None = None
    web_sources = ""
    web_context = ""
    web_backend = "web"
    try:
        web_result, web_notices, answer_text, web_backend = web_search(
            query,
            config,
            loop_count,
        )
        notices.extend(web_notices)
        if web_result and web_result.get("results"):
            web_sources, web_context = prepare_research_context(
                web_result,
                answer_text,
                config,
            )
    except Exception as exc:  # noqa: BLE001 - degradation boundary
        notices.append(f"Web Search 不可用，继续使用本地 Knowledge Evidence：{exc}")

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
    )


def _format_knowledge(results: list[RetrievalResult]) -> tuple[str, str]:
    source_lines: list[str] = []
    contexts: list[str] = []
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
    return "\n".join(source_lines), "\n\n".join(contexts)
