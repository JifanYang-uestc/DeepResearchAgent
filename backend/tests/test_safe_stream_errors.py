"""SSE failure events must not expose raw exceptions to clients."""

from __future__ import annotations

from threading import Lock

from fastapi.testclient import TestClient

import main as main_module
from agent import DeepResearchAgent
from config import Configuration
from models import TodoItem

SENSITIVE_ERROR = r"secret-token=abc123 path=C:\Users\test\model"


class OneTaskPlanner:
    def __init__(self) -> None:
        self.task = TodoItem(id=1, title="test", intent="test", query="test")

    def plan_todo_list(self, state):
        return [self.task]


class SafeReporting:
    def generate_report(self, state):
        return "# Partial report"


class FakeTracker:
    def set_event_sink(self, sink):
        self.sink = sink

    def drain(self, state, step=None):
        return []

    def as_dicts(self):
        return []


def test_task_failure_sse_does_not_leak_raw_exception(caplog) -> None:
    def failed_execute(*args, **kwargs):
        raise RuntimeError(SENSITIVE_ERROR)
        yield  # pragma: no cover - make this a generator

    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.config = Configuration(enable_notes=False)
    planner = OneTaskPlanner()
    agent.planner = planner
    agent.reporting = SafeReporting()
    agent.note_tool = None
    agent._tool_tracker = FakeTracker()
    agent._tool_event_sink_enabled = False
    agent._state_lock = Lock()
    agent._execute_task = failed_execute

    events = list(agent.run_stream("safe failure"))
    failed = next(
        event
        for event in events
        if event.get("type") == "task_status" and event.get("status") == "failed"
    )

    assert "abc123" not in failed["detail"]
    assert r"C:\Users\test\model" not in failed["detail"]
    assert planner.task.status == "failed"
    assert "abc123" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_outer_stream_error_does_not_leak_raw_exception(monkeypatch) -> None:
    class FailingStreamingAgent:
        def __init__(self, **kwargs) -> None:
            self.config = kwargs.get("config")

        def run_stream(self, topic: str):
            raise RuntimeError(SENSITIVE_ERROR)
            yield  # pragma: no cover - make this a generator

    monkeypatch.setattr(main_module, "DeepResearchAgent", FailingStreamingAgent)
    with TestClient(main_module.create_app()) as client:
        with client.stream(
            "POST",
            "/research/stream",
            json={"topic": "safe failure"},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "abc123" not in body
    assert r"C:\Users\test\model" not in body
    assert "研究流程暂时不可用" in body
