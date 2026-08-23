"""State models used by the deep research workflow."""

import operator
from dataclasses import dataclass, field
from typing import Any, List, Optional

from typing_extensions import Annotated


@dataclass(kw_only=True)
class TodoItem:
    """单个待办任务项。"""

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: Optional[str] = field(default=None)
    sources_summary: Optional[str] = field(default=None)
    source_items: list[dict[str, Any]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    retrieval_route: Optional[str] = field(default=None)
    retrieval_reason: Optional[str] = field(default=None)
    retrieval_confidence: Optional[float] = field(default=None)
    freshness_required: bool = field(default=False)
    retrieval_metrics_ms: dict[str, float | None] = field(default_factory=dict)
    note_id: Optional[str] = field(default=None)
    note_path: Optional[str] = field(default=None)
    stream_token: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # Report topic
    research_mode: str = field(default="web")
    document_set_id: Optional[str] = field(default=None)
    search_query: str = field(default=None)  # Deprecated placeholder
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    running_summary: str = field(default=None)  # Legacy summary field
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: Optional[str] = field(default=None)
    report_note_id: Optional[str] = field(default=None)
    report_note_path: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)  # Backward-compatible文本
    report_markdown: Optional[str] = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)

