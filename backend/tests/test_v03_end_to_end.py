"""Deterministic local end-to-end SSE acceptance for all V0.3 routes."""

from __future__ import annotations

from threading import Lock

import agent as agent_module
from agent import DeepResearchAgent
from config import Configuration
from models import TodoItem
from rag.base import KnowledgeDocumentInfo
from rag.types import KnowledgeChunk, RetrievalResult
from services.relevance_gate import KnowledgeRelevanceGate
from services.research import gather_research_evidence as real_gather
from services.retrieval_router import RetrievalRouter


class FakePlanner:
    def plan_todo_list(self, state):
        return [
            TodoItem(  # type: ignore[call-arg]
                id=1,
                title="ReAct 原理",
                intent="解释 ReAct reasoning and acting",
                query="ReAct reasoning and acting 原理",
            ),
            TodoItem(  # type: ignore[call-arg]
                id=2,
                title="机器人趋势",
                intent="调查当前机器人趋势",
                query="机器人领域当前的发展趋势",
            ),
            TodoItem(  # type: ignore[call-arg]
                id=3,
                title="Agentic RAG 趋势",
                intent="结合 RAG 基础与最新资料",
                query="RAG 与 2026 Agentic RAG 最新趋势",
            ),
        ]


class FakeKnowledge:
    def get_catalog(self):
        return [
            KnowledgeDocumentInfo("rag_2020.pdf", "pdf", 19),
            KnowledgeDocumentInfo("react.pdf", "pdf", 33),
            KnowledgeDocumentInfo("self_rag.pdf", "pdf", 30),
        ], []

    def retrieve_with_backend(self, query: str, *, top_k: int):
        document = "react.pdf" if "react" in query.lower() else "rag_2020.pdf"
        content = f"Relevant semantic evidence for {query}"
        chunk = KnowledgeChunk(
            chunk_id=f"{document}-chunk",
            content=content,
            document=document,
            source_path=document,
            file_type="pdf",
            page=2,
            start_char=0,
            end_char=len(content),
        )
        return [RetrievalResult(1, 0.8, chunk)], [], "helloagents"


class FakeSummarizer:
    def stream_task_summary(self, state, task, context):
        summary = f"任务总结：{task.title}"
        return iter([summary]), lambda: summary


class FakeReporting:
    def generate_report(self, state):
        return "# Report\n[Knowledge] react.pdf\n[Web] https://example.com/current"


class FakeTracker:
    def __init__(self) -> None:
        self.sink = None

    def set_event_sink(self, sink):
        self.sink = sink

    def drain(self, state, step=None):
        return []

    def as_dicts(self):
        return []


def _fake_web(*args: object):
    return (
        {
            "results": [
                {
                    "title": "Current evidence",
                    "url": "https://example.com/current",
                    "content": "Current web evidence.",
                }
            ]
        },
        [],
        None,
        "tavily",
    )


def test_local_sse_flow_covers_all_routes_and_done(monkeypatch) -> None:
    def fake_gather(query, config, loop_count, knowledge, **kwargs):
        return real_gather(
            query,
            config,
            loop_count,
            knowledge,
            web_search=_fake_web,
            **kwargs,
        )

    monkeypatch.setattr(agent_module, "gather_research_evidence", fake_gather)
    config = Configuration(enable_notes=False, fetch_full_page=False)
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.config = config
    agent.planner = FakePlanner()
    agent.knowledge = FakeKnowledge()
    agent.router = RetrievalRouter(config)
    agent.relevance_gate = KnowledgeRelevanceGate(config)
    agent.summarizer = FakeSummarizer()
    agent.reporting = FakeReporting()
    agent.note_tool = None
    agent._tool_tracker = FakeTracker()
    agent._tool_event_sink_enabled = False
    agent._state_lock = Lock()
    agent._last_search_notices = []

    events = list(agent.run_stream("V0.3 route acceptance"))
    routes = {
        event["task_id"]: event["route"]
        for event in events
        if event.get("type") == "retrieval_route"
    }
    source_types = {
        source["type"]
        for event in events
        if event.get("type") == "sources"
        for source in event.get("sources", [])
    }

    assert routes == {1: "knowledge", 2: "web", 3: "hybrid"}
    assert source_types == {"knowledge", "web"}
    assert any(event.get("type") == "final_report" for event in events)
    assert events[-1] == {"type": "done"}
