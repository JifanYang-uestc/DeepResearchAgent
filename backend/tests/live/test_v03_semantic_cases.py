"""Local-model V0.3 Knowledge/Web/Hybrid acceptance cases."""

from __future__ import annotations

import os

import pytest

from config import Configuration
from models import TodoItem
from services.knowledge import KnowledgeService
from services.research import gather_research_evidence
from services.retrieval_router import RetrievalRoute, RetrievalRouter

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_SEMANTIC_LIVE") != "1",
        reason="set RUN_SEMANTIC_LIVE=1 to load the local semantic model",
    ),
]


def _web(*args: object):
    return (
        {
            "results": [
                {
                    "title": "Current Agentic RAG trend",
                    "url": "https://example.com/agentic-rag",
                    "content": "Current web evidence for the acceptance test.",
                }
            ]
        },
        [],
        None,
        "tavily",
    )


def _task(query: str) -> TodoItem:
    return TodoItem(id=1, title=query[:20], intent=query, query=query)


def test_real_semantic_knowledge_web_and_hybrid_cases() -> None:
    config = Configuration(
        knowledge_backend="helloagents",
        knowledge_probe_top_k=3,
        knowledge_relevance_threshold=0.55,
        fetch_full_page=False,
    )
    knowledge = KnowledgeService(config)
    catalog, notices = knowledge.get_catalog()
    assert not notices
    router = RetrievalRouter(config)

    knowledge_query = "ReAct 如何通过 reasoning 和 acting 与环境交互？"
    knowledge_decision = router.route(
        research_topic=knowledge_query,
        task=_task(knowledge_query),
        knowledge_catalog=catalog,
    )
    knowledge_bundle = gather_research_evidence(
        knowledge_query,
        config,
        0,
        knowledge,
        decision=knowledge_decision,
        web_search=lambda *args: (_ for _ in ()).throw(
            AssertionError("Knowledge acceptance must not call Web")
        ),
    )
    assert knowledge_decision.route is RetrievalRoute.KNOWLEDGE
    assert knowledge_bundle.knowledge_results
    assert knowledge_bundle.knowledge_results[0].chunk.document == "react.pdf"
    assert knowledge_bundle.knowledge_results[0].chunk.page is not None

    web_query = "机器人领域的发展趋势"
    web_decision = router.route(
        research_topic=web_query,
        task=_task(web_query),
        knowledge_catalog=catalog,
    )
    web_bundle = gather_research_evidence(
        web_query,
        config,
        0,
        knowledge,
        decision=web_decision,
        web_search=_web,
    )
    assert web_decision.route is RetrievalRoute.WEB
    assert all(source.type == "web" for source in web_bundle.sources)

    hybrid_query = (
        "从传统 RAG 到 Agentic RAG：对比 RAG、ReAct 和 Self-RAG，"
        "并结合 2026 年最新互联网资料分析发展趋势。"
    )
    hybrid_decision = router.route(
        research_topic=hybrid_query,
        task=_task(hybrid_query),
        knowledge_catalog=catalog,
    )
    hybrid_bundle = gather_research_evidence(
        hybrid_query,
        config,
        0,
        knowledge,
        decision=hybrid_decision,
        web_search=_web,
    )
    assert hybrid_decision.route is RetrievalRoute.HYBRID
    assert any(source.type == "knowledge" for source in hybrid_bundle.sources)
    assert any(source.type == "web" for source in hybrid_bundle.sources)
