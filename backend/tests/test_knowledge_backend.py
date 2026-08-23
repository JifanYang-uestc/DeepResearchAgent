"""Knowledge backend abstraction and legacy regression tests."""

from __future__ import annotations

from pathlib import Path

from config import Configuration
from rag.base import KnowledgeBackend
from rag.legacy_faiss_backend import LegacyFaissBackend
from services.knowledge import KnowledgeService


def test_legacy_backend_satisfies_protocol_and_catalog(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "facts.txt").write_text(
        "ResearchX-2026 使用 FAISS 作为 V1 Vector Store。",
        encoding="utf-8",
    )
    config = Configuration(
        knowledge_base_path=str(knowledge_base),
        knowledge_index_path=str(tmp_path / "index"),
        knowledge_chunk_size=200,
        knowledge_chunk_overlap=20,
    )
    backend = LegacyFaissBackend(config)

    assert isinstance(backend, KnowledgeBackend)
    assert backend.health_check()
    assert backend.retrieve("ResearchX-2026 Vector Store", 1)
    assert [entry.document for entry in backend.get_catalog()] == ["facts.txt"]


def test_knowledge_service_accepts_backend_boundary(tmp_path: Path) -> None:
    config = Configuration(
        knowledge_base_path=str(tmp_path / "missing"),
        knowledge_index_path=str(tmp_path / "missing-index"),
    )
    backend = LegacyFaissBackend(config)
    service = KnowledgeService(config, backend=backend)

    assert service.backend_name == "legacy_faiss"


def test_legacy_rebuild_replaces_index_from_current_corpus(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    first_document = knowledge_base / "alpha.txt"
    first_document.write_text("Alpha legacy corpus marker.", encoding="utf-8")
    config = Configuration(
        knowledge_backend="legacy_faiss",
        knowledge_base_path=str(knowledge_base),
        knowledge_index_path=str(tmp_path / "index"),
        knowledge_chunk_size=200,
        knowledge_chunk_overlap=20,
    )
    service = KnowledgeService(config)
    first = service.rebuild()
    first_document.unlink()
    (knowledge_base / "beta.txt").write_text(
        "Beta replacement legacy marker.", encoding="utf-8"
    )

    second = service.rebuild()

    assert first.document_count == second.document_count == 1
    assert second.backend == "legacy_faiss"
    assert service.retrieve("Beta replacement")[0][0].chunk.document == "beta.txt"
