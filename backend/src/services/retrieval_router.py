"""Catalog-aware adaptive routing across knowledge, web, and hybrid search."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Protocol

from config import Configuration
from models import TodoItem
from rag.base import KnowledgeDocumentInfo

logger = logging.getLogger(__name__)

_CATALOG_CONCEPT_ALIASES = (
    ("robotics", "robot", "机器人"),
    ("humanoid", "人形"),
    ("industry", "行业", "产业"),
    ("market", "市场"),
    ("report", "报告"),
)


class RetrievalRoute(str, Enum):
    """Available evidence routes for one research task."""

    KNOWLEDGE = "knowledge"
    WEB = "web"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Typed, observable retrieval decision."""

    route: RetrievalRoute
    reason: str
    confidence: float
    knowledge_query: str | None
    web_query: str | None
    freshness_required: bool

    def __post_init__(self) -> None:
        """Validate routing confidence and observability fields."""

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Routing confidence must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("Routing decision requires a reason")


class RouterDecisionProvider(Protocol):
    """Minimal structured-LLM surface used by the router."""

    def run(self, prompt: str) -> str:
        """Return a JSON routing decision."""


class AgentFactoryRouterProvider:
    """Create an isolated router agent for each concurrent TODO decision."""

    def __init__(self, factory: Callable[[], RouterDecisionProvider]) -> None:
        self._factory = factory

    def run(self, prompt: str) -> str:
        """Run and dispose one short-lived structured routing conversation."""

        agent = self._factory()
        try:
            return agent.run(prompt)
        finally:
            clear_history = getattr(agent, "clear_history", None)
            if callable(clear_history):
                clear_history()


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    """Task-level hard boundaries plus global context for LLM routing."""

    task_freshness_required: bool
    global_freshness_context: bool
    explicit_local_request: bool
    matched_documents: tuple[str, ...]


class RetrievalRouter:
    """Combine deterministic boundaries with optional structured LLM routing."""

    def __init__(
        self,
        config: Configuration,
        decision_provider: RouterDecisionProvider | None = None,
    ) -> None:
        self._config = config
        self._decision_provider = decision_provider

    def route(
        self,
        *,
        research_topic: str,
        task: TodoItem,
        knowledge_catalog: list[KnowledgeDocumentInfo],
        current_date: str | None = None,
    ) -> RoutingDecision:
        """Choose a route using all task context and the current KB catalog."""

        current_date = current_date or date.today().isoformat()
        signals = _collect_signals(research_topic, task, knowledge_catalog)
        fallback = _deterministic_decision(task, signals, knowledge_catalog)

        if not self._config.enable_retrieval_router:
            return RoutingDecision(
                route=RetrievalRoute.HYBRID,
                reason="Adaptive Retrieval Router 已禁用，保持 V0.2 双路检索。",
                confidence=1.0,
                knowledge_query=task.query,
                web_query=task.query,
                freshness_required=signals.task_freshness_required,
            )

        if self._decision_provider is None:
            return fallback

        prompt = _build_router_prompt(
            research_topic=research_topic,
            task=task,
            current_date=current_date,
            knowledge_catalog=knowledge_catalog,
            signals=signals,
        )
        try:
            raw = self._decision_provider.run(prompt)
            decision = _parse_decision(raw, task, signals)
            return _enforce_hard_boundaries(decision, fallback, signals)
        except Exception as exc:  # noqa: BLE001 - deterministic fallback boundary
            logger.warning("Retrieval router failed; using deterministic fallback: %s", exc)
            return RoutingDecision(
                route=fallback.route,
                reason=f"Router 不可用，使用确定性回退：{fallback.reason}",
                confidence=fallback.confidence,
                knowledge_query=fallback.knowledge_query,
                web_query=fallback.web_query,
                freshness_required=fallback.freshness_required,
            )


def _collect_signals(
    research_topic: str,
    task: TodoItem,
    catalog: list[KnowledgeDocumentInfo],
) -> RoutingSignals:
    task_text = _normalize(" ".join((task.title, task.intent, task.query)))
    global_text = _normalize(research_topic)
    freshness_terms = (
        "最新",
        "当前",
        "目前",
        "今日",
        "近期",
        "趋势",
        "市场",
        "融资",
        "价格",
        "政策变化",
        "新闻",
        "latest",
        "current",
        "today",
        "recent",
        "trend",
        "market",
        "funding",
        "news",
    )
    local_terms = (
        "根据本地文档",
        "基于本地文档",
        "根据知识库",
        "基于知识库",
        "local document",
        "knowledge base",
    )
    matched = tuple(
        entry.document
        for entry in catalog
        if _document_matches(task_text, entry)
    )
    explicit_filename = any(
        _normalize(entry.document) in task_text for entry in catalog
    )
    return RoutingSignals(
        task_freshness_required=any(term in task_text for term in freshness_terms),
        global_freshness_context=any(term in global_text for term in freshness_terms),
        explicit_local_request=explicit_filename
        or any(term in task_text for term in local_terms),
        matched_documents=matched,
    )


def _document_matches(text: str, entry: KnowledgeDocumentInfo) -> bool:
    stem = _normalize(Path(entry.document).stem)
    aliases = {stem, stem.replace("_", " ").replace("-", " ")}
    aliases.update(
        part
        for part in re.split(r"[^a-z0-9\u3400-\u9fff]+", stem)
        if len(part) >= 3 and not part.isdigit()
    )
    title = _normalize(entry.title or "")
    if title:
        aliases.add(title)
        aliases.update(
            part
            for part in re.split(r"[^a-z0-9\u3400-\u9fff]+", title)
            if len(part) >= 3 and not part.isdigit()
        )
        for concept_group in _CATALOG_CONCEPT_ALIASES:
            if any(alias in title for alias in concept_group):
                aliases.update(concept_group)
    return any(
        alias
        and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text)
        for alias in aliases
    )


def _deterministic_decision(
    task: TodoItem,
    signals: RoutingSignals,
    catalog: list[KnowledgeDocumentInfo],
) -> RoutingDecision:
    query = task.query.strip()
    has_catalog_match = bool(signals.matched_documents)
    if signals.task_freshness_required and (
        has_catalog_match or signals.explicit_local_request
    ):
        documents = ", ".join(signals.matched_documents) or "本地知识库"
        return RoutingDecision(
            route=RetrievalRoute.HYBRID,
            reason=f"{documents} 可提供基础资料，同时任务要求当前或最新信息。",
            confidence=0.95,
            knowledge_query=query,
            web_query=query,
            freshness_required=True,
        )
    if signals.task_freshness_required and not has_catalog_match:
        catalog_names = ", ".join(item.document for item in catalog) or "空"
        return RoutingDecision(
            route=RetrievalRoute.WEB,
            reason=f"任务要求时效性信息，且 Knowledge Catalog（{catalog_names}）无直接匹配文档。",
            confidence=0.96,
            knowledge_query=None,
            web_query=query,
            freshness_required=True,
        )
    if has_catalog_match or signals.explicit_local_request:
        documents = ", ".join(signals.matched_documents) or "本地知识库"
        return RoutingDecision(
            route=RetrievalRoute.KNOWLEDGE,
            reason=f"任务直接涉及 {documents}，且不要求实时信息。",
            confidence=0.94,
            knowledge_query=query,
            web_query=None,
            freshness_required=False,
        )
    return RoutingDecision(
        route=RetrievalRoute.HYBRID,
        reason="没有足够强的单路信号，使用 Hybrid 并由 Knowledge Relevance Gate 过滤本地候选。",
        confidence=0.55,
        knowledge_query=query,
        web_query=query,
        freshness_required=False,
    )


def _build_router_prompt(
    *,
    research_topic: str,
    task: TodoItem,
    current_date: str,
    knowledge_catalog: list[KnowledgeDocumentInfo],
    signals: RoutingSignals,
) -> str:
    catalog = [entry.to_dict() for entry in knowledge_catalog]
    return (
        "你是检索路由器。只输出一个 JSON 对象，不要 Markdown。\n"
        "route 只能是 knowledge、web、hybrid。confidence 必须在 0 到 1。\n"
        "Knowledge 只用于与 Catalog 直接相关且无需新鲜度的任务；Web 用于域外或时效性任务；"
        "Hybrid 用于本地基础资料加最新互联网资料。\n"
        f"当前日期：{current_date}\n"
        f"研究主题：{research_topic}\n"
        f"任务标题：{task.title}\n"
        f"任务意图：{task.intent}\n"
        f"任务查询：{task.query}\n"
        f"Knowledge Catalog：{json.dumps(catalog, ensure_ascii=False)}\n"
        f"确定性信号：task_freshness={signals.task_freshness_required}, "
        f"global_freshness_context={signals.global_freshness_context}, "
        f"explicit_local={signals.explicit_local_request}, "
        f"matched_documents={list(signals.matched_documents)}\n"
        "输出字段：route, reason, confidence, knowledge_query, web_query, freshness_required"
    )


def _parse_decision(
    raw: str,
    task: TodoItem,
    signals: RoutingSignals,
) -> RoutingDecision:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Router did not return a JSON object")
    payload = json.loads(raw[start : end + 1])
    route = RetrievalRoute(str(payload["route"]).lower())
    knowledge_query = payload.get("knowledge_query")
    web_query = payload.get("web_query")
    if route in (RetrievalRoute.KNOWLEDGE, RetrievalRoute.HYBRID):
        knowledge_query = str(knowledge_query or task.query).strip()
    else:
        knowledge_query = None
    if route in (RetrievalRoute.WEB, RetrievalRoute.HYBRID):
        web_query = str(web_query or task.query).strip()
    else:
        web_query = None
    return RoutingDecision(
        route=route,
        reason=str(payload.get("reason") or "LLM structured routing decision").strip(),
        confidence=float(payload.get("confidence", 0.5)),
        knowledge_query=knowledge_query,
        web_query=web_query,
        freshness_required=signals.task_freshness_required,
    )


def _enforce_hard_boundaries(
    decision: RoutingDecision,
    fallback: RoutingDecision,
    signals: RoutingSignals,
) -> RoutingDecision:
    if signals.task_freshness_required and not signals.matched_documents:
        return fallback
    if signals.task_freshness_required and (
        signals.matched_documents or signals.explicit_local_request
    ):
        return fallback
    if signals.matched_documents and not signals.task_freshness_required:
        return fallback
    return decision


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).lower()
