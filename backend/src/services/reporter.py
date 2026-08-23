"""Service that consolidates task results into the final report."""

from __future__ import annotations

import json

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState
from services.text_processing import strip_tool_calls
from services.user_messages import DOCUMENT_EVIDENCE_INSUFFICIENT
from utils import strip_thinking_tokens


class ReportingService:
    """Generates the final structured report."""

    def __init__(self, report_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = report_agent
        self._config = config

    def generate_report(self, state: SummaryState) -> str:
        """Generate a structured report based on completed tasks."""

        if not any(task.source_items for task in state.todo_items):
            if state.research_mode == "document":
                return f"# 资料不足\n\n{DOCUMENT_EVIDENCE_INSUFFICIENT}"
            return "# 资料不足\n\n当前允许的信息源未提供足够证据，未生成研究结论。"

        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "暂无可用信息"
            sources_block = task.sources_summary or "暂无来源"
            tasks_block.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        note_references = []
        for task in state.todo_items:
            if task.note_id:
                note_references.append(
                    f"- 任务 {task.id}《{task.title}》：note_id={task.note_id}"
                )

        notes_section = "\n".join(note_references) if note_references else "- 暂无可用任务笔记"

        read_template = json.dumps({"action": "read", "note_id": "<note_id>"}, ensure_ascii=False)
        create_conclusion_template = json.dumps(
            {
                "action": "create",
                "title": f"研究报告：{state.research_topic}",
                "note_type": "conclusion",
                "tags": ["deep_research", "report"],
                "content": "请在此沉淀最终报告要点",
            },
            ensure_ascii=False,
        )

        if state.research_mode == "document":
            grounding = (
                "报告只能使用 [Document] 来源；禁止使用 Web 或模型自身知识补充事实。"
            )
        elif state.research_mode == "hybrid":
            grounding = "报告必须明确区分 [Document] 与 [Web] 来源。"
        else:
            grounding = "报告只能使用 [Web] 来源。"

        prompt = (
            f"研究主题：{state.research_topic}\n"
            f"信息源约束：{grounding}\n"
            f"任务概览：\n{''.join(tasks_block)}\n"
            f"可用任务笔记：\n{notes_section}\n"
            f"请针对每条任务笔记使用格式：[TOOL_CALL:note:{read_template}] 读取内容，整合所有信息后撰写报告。\n"
            f"如需输出汇总结论，可追加调用：[TOOL_CALL:note:{create_conclusion_template}] 保存报告要点。"
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()

        return report_text or "报告生成失败，请检查输入。"

