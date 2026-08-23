"""Synchronous research must execute every TODO before reporting."""

from __future__ import annotations

from threading import Lock

from fastapi.testclient import TestClient

import agent as agent_module
import main as main_module
from agent import DeepResearchAgent
from config import Configuration
from models import TodoItem
from services.research import EvidenceBundle, EvidenceSource
from services.retrieval_router import RetrievalRoute, RoutingDecision


class TwoTaskPlanner:
    def plan_todo_list(self, state):
        return [
            TodoItem(id=1, title="one", intent="one", query="one"),
            TodoItem(id=2, title="two", intent="two", query="two"),
        ]


class FakeKnowledge:
    def get_catalog(self):
        return [], []


class WebRouter:
    def route(self, **kwargs):
        task = kwargs["task"]
        return RoutingDecision(
            route=RetrievalRoute.WEB,
            reason="test web mode",
            confidence=1.0,
            knowledge_query=None,
            web_query=task.query,
            freshness_required=False,
        )


class SyncSummarizer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def summarize_task(self, state, task, context):
        self.calls.append(task.id)
        return f"summary-{task.id}: {context}"


class CapturingReporter:
    def __init__(self) -> None:
        self.snapshot: list[tuple[str, str | None]] = []

    def generate_report(self, state):
        self.snapshot = [(task.status, task.summary) for task in state.todo_items]
        return "# Report\n" + "\n".join(task.summary or "" for task in state.todo_items)


class FakeTracker:
    def drain(self, state, step=None):
        return []

    def as_dicts(self):
        return []


def _build_agent(monkeypatch) -> tuple[DeepResearchAgent, list[str]]:
    retrieval_calls: list[str] = []

    def fake_gather(query, *args, **kwargs):
        retrieval_calls.append(query)
        return EvidenceBundle(
            context=f"evidence-{query}",
            sources_summary=f"[Web] {query}",
            backend="web",
            sources=[
                EvidenceSource(
                    type="web",
                    title=query,
                    url=f"https://example.com/{query}",
                )
            ],
            timings_ms={"total_retrieval": 1.0},
        )

    monkeypatch.setattr(agent_module, "gather_research_evidence", fake_gather)
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.config = Configuration(enable_notes=False)
    agent.planner = TwoTaskPlanner()
    agent.knowledge = FakeKnowledge()
    agent.router = WebRouter()
    agent.relevance_gate = object()
    agent.summarizer = SyncSummarizer()
    agent.reporting = CapturingReporter()
    agent.note_tool = None
    agent._tool_tracker = FakeTracker()
    agent._tool_event_sink_enabled = False
    agent._state_lock = Lock()
    agent._last_search_notices = []
    return agent, retrieval_calls


def test_sync_run_executes_all_todos(monkeypatch) -> None:
    agent, retrieval_calls = _build_agent(monkeypatch)

    result = agent.run("sync topic")

    assert retrieval_calls == ["one", "two"]
    assert [task.status for task in result.todo_items] == ["completed", "completed"]
    assert [task.summary for task in result.todo_items] == [
        "summary-1: evidence-one",
        "summary-2: evidence-two",
    ]
    assert agent.reporting.snapshot == [
        ("completed", "summary-1: evidence-one"),
        ("completed", "summary-2: evidence-two"),
    ]


def test_post_research_executes_retrieval_and_summary(monkeypatch) -> None:
    agent, retrieval_calls = _build_agent(monkeypatch)
    monkeypatch.setattr(main_module, "DeepResearchAgent", lambda config=None: agent)

    with TestClient(main_module.create_app()) as client:
        response = client.post("/research", json={"topic": "sync endpoint"})

    assert response.status_code == 200
    assert retrieval_calls == ["one", "two"]
    payload = response.json()
    assert [item["status"] for item in payload["todo_items"]] == [
        "completed",
        "completed",
    ]
    assert "summary-1" in payload["report_markdown"]
