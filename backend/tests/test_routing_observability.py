"""SSE routing observability regression tests."""

from __future__ import annotations

from threading import Lock

from agent import DeepResearchAgent
from config import Configuration
from models import SummaryState, TodoItem
from rag.base import KnowledgeDocumentInfo
from rag.types import KnowledgeChunk, RetrievalResult
from services.relevance_gate import KnowledgeRelevanceGate
from services.retrieval_router import RetrievalRoute, RoutingDecision


class FakeTracker:
    def drain(self, state: SummaryState, step: int | None = None):
        return []


class FakeKnowledge:
    def get_catalog(self):
        return [KnowledgeDocumentInfo("react.pdf", "pdf", 33)], []

    def retrieve_with_backend(self, query: str, *, top_k: int):
        chunk = KnowledgeChunk(
            chunk_id="react-2",
            content="ReAct interleaves reasoning and acting.",
            document="react.pdf",
            source_path="react.pdf",
            file_type="pdf",
            page=2,
            start_char=0,
            end_char=40,
        )
        return [RetrievalResult(1, 0.8, chunk)], [], "helloagents"


class FakeRouter:
    def route(self, **kwargs):
        return RoutingDecision(
            route=RetrievalRoute.KNOWLEDGE,
            reason="query directly concerns react.pdf",
            confidence=0.98,
            knowledge_query="ReAct reasoning and acting",
            web_query=None,
            freshness_required=False,
        )


class FakeSummarizer:
    def stream_task_summary(self, state, task, context):
        return iter(["summary"]), lambda: "summary"


def test_execute_task_emits_route_before_sources() -> None:
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.config = Configuration()
    agent.knowledge = FakeKnowledge()
    agent.router = FakeRouter()
    agent.relevance_gate = KnowledgeRelevanceGate(agent.config)
    agent.summarizer = FakeSummarizer()
    agent._tool_tracker = FakeTracker()
    agent._tool_event_sink_enabled = False
    agent._state_lock = Lock()
    agent._last_search_notices = []
    state = SummaryState(research_topic="ReAct 原理")
    task = TodoItem(
        id=1,
        title="ReAct",
        intent="解释原理",
        query="ReAct reasoning and acting",
    )

    events = list(agent._execute_task(state, task, emit_stream=True, step=1))
    event_types = [event["type"] for event in events]

    assert event_types.index("retrieval_route") < event_types.index("sources")
    route_event = next(event for event in events if event["type"] == "retrieval_route")
    assert route_event["route"] == "knowledge"
    assert route_event["confidence"] == 0.98
    assert task.retrieval_metrics_ms["router"] >= 0
    assert task.retrieval_metrics_ms["total_task"] >= 0
