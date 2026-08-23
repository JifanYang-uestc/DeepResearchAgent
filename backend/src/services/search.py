"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Optional, Tuple
from urllib.parse import urlsplit

from hello_agents.tools import SearchTool

from config import Configuration
from services.user_messages import WEB_PROVIDER_NOTICE
from utils import (
    CHARS_PER_TOKEN,
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
MAX_RESULTS = 5
MAX_TEXT_CHARS = MAX_TOKENS_PER_SOURCE * CHARS_PER_TOKEN


@lru_cache(maxsize=1)
def _get_search_tool() -> SearchTool:
    """Create the HelloAgents search client only when Web evidence is requested."""

    return SearchTool(backend="hybrid")


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)

    try:
        raw_response = _get_search_tool().run(
            {
                "input": query,
                "backend": search_api,
                "mode": "structured",
                "fetch_full_page": config.fetch_full_page,
                "max_results": 5,
                "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                "loop_count": loop_count,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise

    payload, response_issue = normalize_search_response(raw_response, search_api)
    notices = [WEB_PROVIDER_NOTICE] if response_issue else []

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def normalize_search_response(
    raw_response: Any,
    fallback_backend: str,
) -> tuple[dict[str, Any], bool]:
    """Validate untrusted search data before it reaches prompts or the frontend."""

    if not isinstance(raw_response, Mapping):
        logger.warning(
            "Search backend %s returned an unsupported payload type: %s",
            fallback_backend,
            type(raw_response).__name__,
        )
        return _empty_search_payload(fallback_backend), True

    response_issue = False
    raw_backend = raw_response.get("backend")
    backend = (
        raw_backend.strip()
        if isinstance(raw_backend, str) and raw_backend.strip()
        else fallback_backend
    )

    raw_answer = raw_response.get("answer")
    if raw_answer is not None and not isinstance(raw_answer, str):
        response_issue = True
    answer = _bounded_text(raw_answer) if isinstance(raw_answer, str) else None

    raw_notices = raw_response.get("notices")
    if raw_notices:
        response_issue = True
        notice_count = len(raw_notices) if isinstance(raw_notices, list) else 1
        logger.warning(
            "Search backend %s returned %s provider notice(s)",
            fallback_backend,
            notice_count,
        )

    raw_results = raw_response.get("results", [])
    if not isinstance(raw_results, list):
        logger.warning(
            "Search backend %s returned non-list results",
            fallback_backend,
        )
        raw_results = []
        response_issue = True

    if len(raw_results) > MAX_RESULTS:
        response_issue = True

    results: list[dict[str, str]] = []
    for item in raw_results[:MAX_RESULTS]:
        normalized = _normalize_search_result(item)
        if normalized is None:
            response_issue = True
            continue
        results.append(normalized)

    return (
        {
            "results": results,
            "backend": backend,
            "answer": answer,
            "notices": [],
        },
        response_issue,
    )


def _normalize_search_result(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None

    url = _safe_web_url(value.get("url"))
    if url is None:
        return None

    title = _bounded_text(value.get("title")) or url
    content = _bounded_text(value.get("content"))
    raw_content = _bounded_text(value.get("raw_content"))
    return {
        "title": title,
        "url": url,
        "content": content,
        "raw_content": raw_content,
    }


def _safe_web_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 2048:
        return None
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None and not 1 <= port <= 65535
    ):
        return None
    return candidate


def _bounded_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:MAX_TEXT_CHARS]


def _empty_search_payload(backend: str) -> dict[str, Any]:
    return {
        "results": [],
        "backend": backend,
        "answer": None,
        "notices": [],
    }


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
