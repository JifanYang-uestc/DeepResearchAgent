"""Knowledge/Web dual-route integration and degradation tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import Configuration
from rag.types import KnowledgeChunk, RetrievalResult
from services.research import gather_research_evidence


@dataclass
class FakeKnowledgeService:
    results: list[RetrievalResult]
    notices: list[str]

    def retrieve(self, query: str) -> tuple[list[RetrievalResult], list[str]]:
        return self.results, self.notices


def _knowledge_result() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=0.91,
        chunk=KnowledgeChunk(
            chunk_id="chunk-local",
            content="Local evidence remains available.",
            document="facts.md",
            source_path=str(Path("facts.md")),
            file_type="md",
            page=None,
            start_char=0,
            end_char=33,
        ),
    )


def test_web_failure_keeps_local_evidence() -> None:
    def failed_web(*args: object) -> object:
        raise RuntimeError("web offline")

    bundle = gather_research_evidence(
        "query",
        Configuration(),
        0,
        FakeKnowledgeService([_knowledge_result()], []),  # type: ignore[arg-type]
        web_search=failed_web,  # type: ignore[arg-type]
    )

    assert bundle.available
    assert bundle.backend == "knowledge"
    assert "Local evidence remains available" in bundle.context
    assert "Web Search 不可用" in bundle.notices[0]


def test_knowledge_failure_keeps_web_evidence() -> None:
    def working_web(*args: object) -> tuple[dict, list[str], None, str]:
        return (
            {
                "results": [
                    {
                        "title": "Web evidence",
                        "url": "https://example.com/evidence",
                        "content": "Current web evidence remains available.",
                    }
                ]
            },
            [],
            None,
            "tavily",
        )

    bundle = gather_research_evidence(
        "query",
        Configuration(fetch_full_page=False),
        0,
        FakeKnowledgeService([], ["knowledge offline"]),  # type: ignore[arg-type]
        web_search=working_web,
    )

    assert bundle.available
    assert bundle.backend == "tavily"
    assert "Current web evidence remains available" in bundle.context
    assert bundle.notices == ["knowledge offline"]


def test_dual_route_combines_evidence_and_sources() -> None:
    def working_web(*args: object) -> tuple[dict, list[str], None, str]:
        return (
            {
                "results": [
                    {
                        "title": "Latest trend",
                        "url": "https://example.com/trend",
                        "content": "Web trend evidence.",
                    }
                ]
            },
            [],
            None,
            "tavily",
        )

    bundle = gather_research_evidence(
        "query",
        Configuration(fetch_full_page=False),
        0,
        FakeKnowledgeService([_knowledge_result()], []),  # type: ignore[arg-type]
        web_search=working_web,
    )

    assert bundle.backend == "knowledge+tavily"
    assert "[Knowledge] facts.md" in bundle.sources_summary
    assert "https://example.com/trend" in bundle.sources_summary
    assert "Knowledge Evidence" in bundle.context
    assert "Web trend evidence" in bundle.context
    assert [source.type for source in bundle.sources] == ["knowledge", "web"]
    assert bundle.sources[0].chunk_id == "chunk-local"
    assert bundle.sources[1].url == "https://example.com/trend"
