"""Knowledge index lifecycle and safe degradation tests."""

from __future__ import annotations

from pathlib import Path

from config import Configuration
from rag.base import KnowledgeDocumentInfo
from rag.types import KnowledgeChunk, RetrievalResult
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
        knowledge_backend="legacy_faiss",
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
        knowledge_backend="legacy_faiss",
        knowledge_base_path=str(tmp_path / "missing"),
        knowledge_index_path=str(tmp_path / "missing-index"),
    )

    results, notices = KnowledgeService(config).retrieve("query")

    assert results == []
    assert len(notices) == 1
    assert "退化到 Web Search" in notices[0]


class FailingBackend:
    name = "helloagents"

    def __init__(self, message: str) -> None:
        self.message = message

    def retrieve(self, query: str, top_k: int):
        raise RuntimeError(self.message)

    def health_check(self) -> bool:
        return False

    def get_catalog(self):
        raise RuntimeError(self.message)


class WorkingLegacyBackend:
    name = "legacy_faiss"

    def retrieve(self, query: str, top_k: int):
        chunk = KnowledgeChunk(
            chunk_id="legacy",
            content="Legacy fallback evidence",
            document="facts.txt",
            source_path="facts.txt",
            file_type="txt",
            page=None,
            start_char=0,
            end_char=24,
        )
        return [RetrievalResult(1, 0.1, chunk)]

    def health_check(self) -> bool:
        return True

    def get_catalog(self):
        return [KnowledgeDocumentInfo("facts.txt", "txt", 1)]


def test_semantic_embedding_failure_falls_back_to_legacy() -> None:
    service = KnowledgeService(
        Configuration(),
        backend=FailingBackend("semantic embedding failure"),
        fallback_backend=WorkingLegacyBackend(),
    )

    results, notices, backend = service.retrieve_with_backend("query")

    assert results
    assert backend == "legacy_faiss"
    assert "本地回退后端" in notices[0]
    assert "semantic embedding failure" not in notices[0]


def test_vector_backend_unavailable_falls_back_to_legacy() -> None:
    service = KnowledgeService(
        Configuration(),
        backend=FailingBackend("vector backend unavailable"),
        fallback_backend=WorkingLegacyBackend(),
    )

    results, notices, backend = service.retrieve_with_backend("query")

    assert results
    assert backend == "legacy_faiss"
    assert "本地回退后端" in notices[0]
    assert "vector backend unavailable" not in notices[0]


def test_user_visible_notice_does_not_leak_raw_exception() -> None:
    sensitive = r"secret-token=abc123 path=C:\Users\test\model"
    service = KnowledgeService(
        Configuration(),
        backend=FailingBackend(sensitive),
        fallback_backend=FailingBackend(sensitive),
    )

    results, notices, backend = service.retrieve_with_backend("query")

    assert results == []
    assert backend == "none"
    assert notices
    assert "abc123" not in notices[0]
    assert r"C:\Users\test\model" not in notices[0]
    assert sensitive not in notices[0]
