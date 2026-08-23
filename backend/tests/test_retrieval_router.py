"""Adaptive retrieval routing matrix and deterministic fallback tests."""

from __future__ import annotations

import pytest

from config import Configuration
from models import TodoItem
from rag.base import KnowledgeDocumentInfo
from services.retrieval_router import RetrievalRoute, RetrievalRouter


CATALOG = [
    KnowledgeDocumentInfo("rag_2020.pdf", "pdf", 19),
    KnowledgeDocumentInfo("react.pdf", "pdf", 33),
    KnowledgeDocumentInfo("self_rag.pdf", "pdf", 30),
]


def _task(query: str) -> TodoItem:
    return TodoItem(id=1, title="研究任务", intent=query, query=query)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ReAct 如何结合 reasoning 和 acting？", RetrievalRoute.KNOWLEDGE),
        ("RAG 中 parametric 与 non-parametric memory 有什么区别？", RetrievalRoute.KNOWLEDGE),
        ("机器人领域的发展趋势", RetrievalRoute.WEB),
        ("2026 年机器人行业最新融资和市场趋势", RetrievalRoute.WEB),
        (
            "对比 RAG、ReAct、Self-RAG 并分析 2026 Agentic RAG 最新趋势",
            RetrievalRoute.HYBRID,
        ),
        (
            "基于 self_rag.pdf 解释反思机制，并结合最新 Agentic RAG 应用",
            RetrievalRoute.HYBRID,
        ),
    ],
)
def test_required_routing_matrix(query: str, expected: RetrievalRoute) -> None:
    decision = RetrievalRouter(Configuration()).route(
        research_topic=query,
        task=_task(query),
        knowledge_catalog=CATALOG,
        current_date="2026-08-23",
    )

    assert decision.route is expected
    assert decision.reason
    assert 0 <= decision.confidence <= 1


def test_robotics_out_of_domain_regression_routes_web() -> None:
    decision = RetrievalRouter(Configuration()).route(
        research_topic="机器人领域的发展趋势",
        task=_task("机器人领域的发展趋势"),
        knowledge_catalog=CATALOG,
    )

    assert decision.route is RetrievalRoute.WEB
    assert decision.knowledge_query is None
    assert decision.web_query


@pytest.mark.parametrize("failure", ["invalid", "timeout"])
def test_router_failure_uses_deterministic_fallback(failure: str) -> None:
    class FailingProvider:
        def run(self, prompt: str) -> str:
            assert "Knowledge Catalog" in prompt
            if failure == "timeout":
                raise TimeoutError("router timeout")
            return "not-json"

    decision = RetrievalRouter(
        Configuration(),
        decision_provider=FailingProvider(),
    ).route(
        research_topic="解释一个尚不明确的研究问题",
        task=_task("解释一个尚不明确的研究问题"),
        knowledge_catalog=CATALOG,
    )

    assert decision.route is RetrievalRoute.HYBRID
    assert "确定性回退" in decision.reason


def test_structured_router_receives_full_context_and_catalog() -> None:
    class CapturingProvider:
        prompt = ""

        def run(self, prompt: str) -> str:
            self.prompt = prompt
            return (
                '{"route":"web","reason":"domain mismatch","confidence":0.8,'
                '"knowledge_query":null,"web_query":"quantum update",'
                '"freshness_required":false}'
            )

    provider = CapturingProvider()
    task = TodoItem(id=2, title="量子任务", intent="调查量子纠错", query="量子纠错")
    decision = RetrievalRouter(Configuration(), provider).route(
        research_topic="量子计算",
        task=task,
        knowledge_catalog=CATALOG,
        current_date="2026-08-23",
    )

    assert decision.route is RetrievalRoute.WEB
    assert "量子计算" in provider.prompt
    assert "量子任务" in provider.prompt
    assert "调查量子纠错" in provider.prompt
    assert "rag_2020.pdf" in provider.prompt
    assert "2026-08-23" in provider.prompt
