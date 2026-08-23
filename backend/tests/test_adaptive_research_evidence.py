"""Route-aware evidence orchestration and failure regression tests."""

from __future__ import annotations

from dataclasses import dataclass

from config import Configuration
from rag.types import KnowledgeChunk, RetrievalResult
from services.research import gather_research_evidence
from services.retrieval_router import RetrievalRoute, RoutingDecision


@dataclass
class TrackingKnowledge:
    results: list[RetrievalResult]
    notices: list[str]
    backend: str = "helloagents"
    calls: int = 0

    def retrieve_with_backend(self, query: str, *, top_k: int):
        self.calls += 1
        return self.results[:top_k], self.notices, self.backend


def _candidate(score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=score,
        chunk=KnowledgeChunk(
            chunk_id="react-chunk",
            content="ReAct interleaves reasoning traces and actions.",
            document="react.pdf",
            source_path="react.pdf",
            file_type="pdf",
            page=2,
            start_char=0,
            end_char=52,
        ),
    )


def _decision(route: RetrievalRoute, query: str) -> RoutingDecision:
    return RoutingDecision(
        route=route,
        reason="test route",
        confidence=1.0,
        knowledge_query=query if route is not RetrievalRoute.WEB else None,
        web_query=query if route is not RetrievalRoute.KNOWLEDGE else None,
        freshness_required=route is not RetrievalRoute.KNOWLEDGE,
    )


def _working_web(*args: object):
    return (
        {
            "results": [
                {
                    "title": "Current source",
                    "url": "https://example.com/current",
                    "content": "Current web evidence.",
                }
            ]
        },
        [],
        None,
        "tavily",
    )


def test_web_route_never_executes_knowledge() -> None:
    knowledge = TrackingKnowledge([_candidate()], [])
    bundle = gather_research_evidence(
        "机器人领域的发展趋势",
        Configuration(fetch_full_page=False),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.WEB, "机器人领域的发展趋势"),
        web_search=_working_web,
    )

    assert knowledge.calls == 0
    assert bundle.knowledge_results == []
    assert [source.type for source in bundle.sources] == ["web"]


def test_knowledge_route_never_executes_web_when_evidence_is_relevant() -> None:
    knowledge = TrackingKnowledge([_candidate()], [])

    def forbidden_web(*args: object):
        raise AssertionError("Web must not be called for a successful knowledge route")

    bundle = gather_research_evidence(
        "ReAct reasoning and acting",
        Configuration(),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.KNOWLEDGE, "ReAct reasoning and acting"),
        web_search=forbidden_web,  # type: ignore[arg-type]
    )

    assert knowledge.calls == 1
    assert [source.type for source in bundle.sources] == ["knowledge"]


def test_hybrid_rejects_weak_knowledge_and_keeps_web() -> None:
    knowledge = TrackingKnowledge([_candidate(0.39)], [])
    bundle = gather_research_evidence(
        "机器人市场趋势",
        Configuration(fetch_full_page=False, knowledge_relevance_threshold=0.55),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.HYBRID, "机器人市场趋势"),
        web_search=_working_web,
    )

    assert bundle.knowledge_results == []
    assert [source.type for source in bundle.sources] == ["web"]
    assert any("Knowledge Evidence 已拒绝" in notice for notice in bundle.notices)


def test_knowledge_failure_falls_back_to_web() -> None:
    knowledge = TrackingKnowledge([], ["semantic embedding failure"], "none")
    bundle = gather_research_evidence(
        "ReAct 原理",
        Configuration(fetch_full_page=False),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.KNOWLEDGE, "ReAct 原理"),
        web_search=_working_web,
    )

    assert bundle.available
    assert [source.type for source in bundle.sources] == ["web"]
    assert any("回退到 Web Search" in notice for notice in bundle.notices)


def test_both_routes_unavailable_returns_no_evidence() -> None:
    knowledge = TrackingKnowledge([], ["vector backend unavailable"], "none")

    def failed_web(*args: object):
        raise RuntimeError("web offline")

    bundle = gather_research_evidence(
        "ambiguous query",
        Configuration(),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.HYBRID, "ambiguous query"),
        web_search=failed_web,  # type: ignore[arg-type]
    )

    assert not bundle.available
    assert bundle.sources == []
    assert any("Web Search 不可用" in notice for notice in bundle.notices)


def test_untrusted_web_payload_is_validated_before_context() -> None:
    knowledge = TrackingKnowledge([], [])

    def untrusted_web(*args: object):
        return (
            {
                "results": [
                    {
                        "title": "Unsafe source",
                        "url": "javascript:alert(1)",
                        "content": "must not reach the summarizer",
                    },
                    {
                        "title": "Safe source",
                        "url": "https://example.com/safe",
                        "content": "Validated web evidence.",
                    },
                ],
                "answer": "Validated direct answer.",
            },
            [],
            "tuple answer must not bypass payload validation",
            "tavily",
        )

    bundle = gather_research_evidence(
        "current topic",
        Configuration(fetch_full_page=False),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=_decision(RetrievalRoute.WEB, "current topic"),
        web_search=untrusted_web,
    )

    assert [source.url for source in bundle.sources] == ["https://example.com/safe"]
    assert "Validated direct answer." in bundle.context
    assert "tuple answer must not bypass" not in bundle.context
    assert "javascript:" not in bundle.context
    assert any("Web Search 返回服务状态提示" in notice for notice in bundle.notices)
