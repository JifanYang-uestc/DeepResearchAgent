"""Agent-level ResearchMode initialization and sync/SSE parity tests."""

from __future__ import annotations

from threading import Lock

import agent as agent_module
from agent import DeepResearchAgent
from config import Configuration
from models import TodoItem
from research_mode import ResearchMode
from services.research import EvidenceBundle, EvidenceSource


class FakeLLMAgent:
    def run(self, prompt: str) -> str:
        return "unused"


class OneTaskPlanner:
    def plan_todo_list(self, state):
        return [TodoItem(id=1, title="task", intent="intent", query="query")]


class ParitySummarizer:
    def summarize_task(self, state, task, context):
        return f"summary: {context}"

    def stream_task_summary(self, state, task, context):
        summary = f"summary: {context}"
        return iter([summary]), lambda: summary


class ParityReporter:
    def generate_report(self, state):
        task = state.todo_items[0]
        return f"# Report\n{task.status}\n{task.summary}\n{task.sources_summary}"


class FakeTracker:
    def __init__(self) -> None:
        self.sink = None

    def set_event_sink(self, sink):
        self.sink = sink

    def drain(self, state, step=None):
        return []

    def as_dicts(self):
        return []


def test_web_agent_does_not_construct_knowledge_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_module.DeepResearchAgent,
        "_init_llm",
        lambda self: object(),
    )
    monkeypatch.setattr(
        agent_module.DeepResearchAgent,
        "_create_tool_aware_agent",
        lambda self, **kwargs: FakeLLMAgent(),
    )
    monkeypatch.setattr(
        agent_module,
        "KnowledgeService",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Knowledge backend must not be initialized")
        ),
    )

    agent = DeepResearchAgent(
        Configuration(enable_notes=False),
        research_mode=ResearchMode.WEB,
    )

    assert agent.knowledge is None
    assert agent.router is None


def _parity_agent() -> DeepResearchAgent:
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.config = Configuration(enable_notes=False)
    agent.research_mode = ResearchMode.WEB
    agent.document_set_id = None
    agent.planner = OneTaskPlanner()
    agent.knowledge = None
    agent.router = None
    agent.relevance_gate = object()
    agent.summarizer = ParitySummarizer()
    agent.reporting = ParityReporter()
    agent.note_tool = None
    agent._tool_tracker = FakeTracker()
    agent._tool_event_sink_enabled = False
    agent._state_lock = Lock()
    agent._last_search_notices = []
    return agent


def test_sync_and_sse_have_matching_task_sources_and_report(monkeypatch) -> None:
    def fake_gather(query, *args, **kwargs):
        return EvidenceBundle(
            context="same evidence",
            sources_summary="[Web] Same : https://example.com/same",
            backend="web",
            sources=[
                EvidenceSource(
                    type="web",
                    title="Same",
                    url="https://example.com/same",
                )
            ],
            timings_ms={"total_retrieval": 1.0},
        )

    monkeypatch.setattr(agent_module, "gather_research_evidence", fake_gather)
    sync_result = _parity_agent().run("same topic")
    stream_events = list(_parity_agent().run_stream("same topic"))

    completed = next(
        event
        for event in stream_events
        if event.get("type") == "task_status" and event.get("status") == "completed"
    )
    final_report = next(
        event["report"]
        for event in stream_events
        if event.get("type") == "final_report"
    )

    assert sync_result.todo_items[0].status == completed["status"] == "completed"
    assert sync_result.todo_items[0].source_items == completed["sources"]
    assert sync_result.report_markdown == final_report
