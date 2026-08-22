"""Knowledge index lifecycle and safe degradation tests."""

from __future__ import annotations

from pathlib import Path

from config import Configuration
from services.knowledge import KnowledgeService


def test_service_auto_builds_and_reuses_index(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "facts.txt").write_text(
        "ResearchX-2026 的测试知识库使用 FAISS 作为 V1 Vector Store。",
        encoding="utf-8",
    )
    index_path = tmp_path / "vector_store"
    config = Configuration(
        knowledge_base_path=str(knowledge_base),
        knowledge_index_path=str(index_path),
        knowledge_chunk_size=200,
        knowledge_chunk_overlap=20,
    )

    first = KnowledgeService(config)
    first_results, first_notices = first.retrieve("ResearchX-2026 Vector Store")
    second = KnowledgeService(config)
    second_results, second_notices = second.retrieve("ResearchX-2026 Vector Store")

    assert first_notices == second_notices == []
    assert first_results[0].chunk.document == "facts.txt"
    assert second_results[0].chunk.chunk_id == first_results[0].chunk.chunk_id
    assert (index_path / "knowledge.faiss").is_file()
    assert (index_path / "metadata.json").is_file()


def test_missing_knowledge_base_degrades_without_raising(tmp_path: Path) -> None:
    config = Configuration(
        knowledge_base_path=str(tmp_path / "missing"),
        knowledge_index_path=str(tmp_path / "missing-index"),
    )

    results, notices = KnowledgeService(config).retrieve("query")

    assert results == []
    assert len(notices) == 1
    assert "退化到 Web Search" in notices[0]
