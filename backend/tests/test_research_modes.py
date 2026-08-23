"""User-controlled source permissions override task-level routing."""

from __future__ import annotations

from dataclasses import dataclass

from config import Configuration
from rag.types import KnowledgeChunk, RetrievalResult
from research_mode import ResearchMode
from services.research import gather_research_evidence
from services.retrieval_router import RetrievalRoute, RoutingDecision


@dataclass
class CountingKnowledge:
    results: list[RetrievalResult]
    calls: int = 0

    def retrieve_with_backend(self, query: str, *, top_k: int):
        self.calls += 1
        return self.results[:top_k], [], "helloagents"


def _candidate(score: float = 0.9) -> RetrievalResult:
    content = "ReAct combines reasoning traces and acting."
    return RetrievalResult(
        rank=1,
        score=score,
        chunk=KnowledgeChunk(
            chunk_id="react-1",
            content=content,
            document="react.pdf",
            source_path="react.pdf",
            file_type="pdf",
            page=1,
            start_char=0,
            end_char=len(content),
        ),
    )


def _decision(route: RetrievalRoute, query: str) -> RoutingDecision:
    return RoutingDecision(
        route=route,
        reason="optimizer suggestion",
        confidence=1.0,
        knowledge_query=query,
        web_query=query,
        freshness_required=True,
    )


def _web(counter: list[str]):
    def search(query, *args):
        counter.append(query)
        return (
            {
                "results": [
                    {
                        "title": "Current source",
                        "url": "https://example.com/current",
                        "content": "Current evidence.",
                    }
                ]
            },
            [],
            None,
            "tavily",
        )

    return search


def test_web_mode_never_calls_document_even_when_optimizer_says_knowledge() -> None:
    knowledge = CountingKnowledge([_candidate()])
    web_calls: list[str] = []

    bundle = gather_research_evidence(
        "ReAct",
        Configuration(fetch_full_page=False),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.KNOWLEDGE, "ReAct"),
        mode=ResearchMode.WEB,
        web_search=_web(web_calls),
    )

    assert knowledge.calls == 0
    assert web_calls == ["ReAct"]
    assert [source.type for source in bundle.sources] == ["web"]


def test_document_mode_never_calls_web_even_for_fresh_query() -> None:
    knowledge = CountingKnowledge([_candidate()])
    web_calls: list[str] = []

    bundle = gather_research_evidence(
        "ReAct 2026 最新应用",
        Configuration(),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.HYBRID, "ReAct 2026 最新应用"),
        mode=ResearchMode.DOCUMENT,
        web_search=_web(web_calls),
    )

    assert knowledge.calls == 1
    assert web_calls == []
    assert [source.type for source in bundle.sources] == ["knowledge"]


def test_document_mode_reports_insufficient_evidence_without_web_fallback() -> None:
    knowledge = CountingKnowledge([_candidate(0.2)])
    web_calls: list[str] = []

    bundle = gather_research_evidence(
        "2026 全球机器人融资趋势",
        Configuration(knowledge_relevance_threshold=0.55),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.WEB, "2026 全球机器人融资趋势"),
        mode=ResearchMode.DOCUMENT,
        web_search=_web(web_calls),
    )

    assert not bundle.available
    assert web_calls == []
    assert bundle.sources == []
    assert any("未提供足够证据" in notice for notice in bundle.notices)


def test_hybrid_mode_runs_document_and_web_with_gate() -> None:
    knowledge = CountingKnowledge([_candidate()])
    web_calls: list[str] = []

    bundle = gather_research_evidence(
        "Agentic RAG 2026",
        Configuration(fetch_full_page=False),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.WEB, "Agentic RAG 2026"),
        mode=ResearchMode.HYBRID,
        web_search=_web(web_calls),
    )

    assert knowledge.calls == 1
    assert web_calls == ["Agentic RAG 2026"]
    assert {source.type for source in bundle.sources} == {"knowledge", "web"}
    assert bundle.gate_result is not None
