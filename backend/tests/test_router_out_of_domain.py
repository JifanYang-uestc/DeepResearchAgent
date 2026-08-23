"""Critical V0.3 robotics-domain contamination regression."""

from __future__ import annotations

from dataclasses import dataclass

from config import Configuration
from models import TodoItem
from rag.base import KnowledgeDocumentInfo
from rag.types import KnowledgeChunk, RetrievalResult
from services.research import gather_research_evidence
from services.retrieval_router import RetrievalRoute, RetrievalRouter

CATALOG = [
    KnowledgeDocumentInfo("rag_2020.pdf", "pdf", 19),
    KnowledgeDocumentInfo("react.pdf", "pdf", 33),
    KnowledgeDocumentInfo("self_rag.pdf", "pdf", 30),
]


@dataclass
class ForbiddenKnowledge:
    calls: int = 0

    def retrieve_with_backend(self, query: str, *, top_k: int):
        self.calls += 1
        chunk = KnowledgeChunk(
            chunk_id="unrelated",
            content="ReAct and RAG are not robotics market evidence.",
            document="react.pdf",
            source_path="react.pdf",
            file_type="pdf",
            page=1,
            start_char=0,
            end_char=48,
        )
        return [RetrievalResult(1, 0.39, chunk)], [], "helloagents"


def test_robotics_query_has_no_knowledge_sources() -> None:
    query = "机器人领域的发展趋势"
    task = TodoItem(id=1, title="机器人趋势", intent=query, query=query)
    decision = RetrievalRouter(Configuration()).route(
        research_topic=query,
        task=task,
        knowledge_catalog=CATALOG,
        current_date="2026-08-23",
    )
    knowledge = ForbiddenKnowledge()

    def web_search(*args: object):
        return (
            {
                "results": [
                    {
                        "title": "Robotics trend",
                        "url": "https://example.com/robotics",
                        "content": "Current robotics evidence.",
                    }
                ]
            },
            [],
            None,
            "tavily",
        )

    bundle = gather_research_evidence(
        query,
        Configuration(fetch_full_page=False),
        0,
        knowledge,  # type: ignore[arg-type]
        decision=decision,
        web_search=web_search,
    )

    assert decision.route is RetrievalRoute.WEB
    assert knowledge.calls == 0
    assert bundle.knowledge_results == []
    assert all(source.type == "web" for source in bundle.sources)
    assert "[Document]" not in bundle.sources_summary

