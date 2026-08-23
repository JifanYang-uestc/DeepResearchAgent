"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState, TodoItem
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)

class PlanningService:
    """Wraps the planner agent to produce structured TODO items."""

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            note_content = str(item.get("content") or "").strip()
            intent = str(
                item.get("intent")
                or note_content
                or "聚焦主题的关键问题"
            ).strip()
            query = str(item.get("query") or "").strip()

            # Tool-aware planners can return the task notes they just created
            # instead of repeating the final TODO schema. Preserve the useful
            # task boundaries and derive a distinct retrieval query from each
            # note rather than falling back every task to the full user topic.
            if not query and note_content:
                query = f"{title} {note_content[:240]}".strip()

            if not query:
                query = state.research_topic

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""

        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""

        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        if not tasks:
            tool_payload = self._extract_tool_payload(text)
            if tool_payload and isinstance(tool_payload.get("tasks"), list):
                for item in tool_payload["tasks"]:
                    if isinstance(item, dict):
                        tasks.append(item)

        if not tasks:
            tasks.extend(self._extract_note_task_payloads(text))

        return tasks

    def _extract_note_task_payloads(self, text: str) -> List[dict[str, Any]]:
        """Recover TODOs from individual note tool calls with nested JSON arrays."""

        marker = "[TOOL_CALL:note:"
        lowered = text.lower()
        decoder = json.JSONDecoder()
        tasks: List[dict[str, Any]] = []
        seen_ids: set[str] = set()
        cursor = 0

        while True:
            marker_index = lowered.find(marker.lower(), cursor)
            if marker_index == -1:
                break
            payload_start = marker_index + len(marker)
            try:
                payload, consumed = decoder.raw_decode(text[payload_start:].lstrip())
            except json.JSONDecodeError:
                cursor = payload_start
                continue
            cursor = payload_start + consumed

            if not isinstance(payload, dict):
                continue
            if payload.get("note_type") != "task_state" or not payload.get("title"):
                continue
            task_identity = str(payload.get("task_id") or payload.get("title"))
            if task_identity in seen_ids:
                continue
            seen_ids.add(task_identity)
            tasks.append(payload)

        return tasks

    def _extract_json_payload(self, text: str) -> Optional[dict[str, Any] | list]:
        """Try to locate and parse a JSON object or array from the text."""

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _extract_tool_payload(self, text: str) -> Optional[dict[str, Any]]:
        """Parse the first TOOL_CALL expression in the output."""

        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return None

        body = match.group("body")

        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")

        return payload or None
