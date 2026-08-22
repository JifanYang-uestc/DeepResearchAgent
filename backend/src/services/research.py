"""Dual-route evidence collection for Knowledge RAG and Web Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from config import Configuration
from rag.types import RetrievalResult
from services.knowledge import KnowledgeService
from services.search import dispatch_search, prepare_research_context

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
    knowledge_sources, knowledge_context, structured_knowledge = _format_knowledge(
        knowledge_results
    )

    web_result: dict[str, Any] | None = None
    web_sources = ""
    web_context = ""
    web_backend = "web"
    structured_web: list[EvidenceSource] = []
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
            structured_web = _web_sources(web_result)
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
        sources=[*structured_knowledge, *structured_web],
    )


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
