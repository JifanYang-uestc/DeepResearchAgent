"""Planner recovery tests for tool-generated task note payloads."""

from __future__ import annotations

from config import Configuration
from models import SummaryState
from services.planner import PlanningService


class FakePlannerAgent:
    def run(self, prompt: str) -> str:
        return """[
          {
            "task_id": 1,
            "title": "任务1: 传统RAG基础",
            "note_type": "task_state",
            "content": "研究 RAG 的参数化与非参数化记忆。"
          },
          {
            "task_id": 2,
            "title": "任务2: ReAct范式",
            "note_type": "task_state",
            "content": "研究 ReAct 的 reasoning 与 acting 交替机制。"
          }
        ]"""

    def clear_history(self) -> None:
        return None


def test_note_payloads_produce_distinct_retrieval_queries() -> None:
    planner = PlanningService(FakePlannerAgent(), Configuration())  # type: ignore[arg-type]
    state = SummaryState(research_topic="对比 RAG 与 ReAct")

    tasks = planner.plan_todo_list(state)

    assert len(tasks) == 2
    assert tasks[0].query != tasks[1].query
    assert "传统RAG基础" in tasks[0].query
    assert "ReAct范式" in tasks[1].query
