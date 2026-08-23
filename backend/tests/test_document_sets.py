"""Uploaded document-set safety, persistence, and isolation tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
from config import Configuration
from rag.base import KnowledgeBuildResult
from rag.catalog import build_knowledge_catalog
from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.types import KnowledgeChunk, RetrievalResult
from services.document_sets import (
    DocumentSetService,
    DocumentUpload,
    DocumentValidationError,
    sanitize_filename,
)


class FakePersistedBackend:
    name = "helloagents"

    def __init__(self, config: Configuration, tracker: "FakeBackendFactory") -> None:
        self.config = config
        self.tracker = tracker
        self.index_path = Path(config.knowledge_index_path) / "fake-index.json"
        self._chunks: list[KnowledgeChunk] | None = None

    def rebuild(self) -> KnowledgeBuildResult:
        pages = load_documents(self.config.knowledge_base_path)
        chunks = chunk_documents(
            pages,
            chunk_size=self.config.knowledge_chunk_size,
            chunk_overlap=self.config.knowledge_chunk_overlap,
        )
        if not chunks:
            raise ValueError("empty index")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        self._chunks = chunks
        self.tracker.rebuilds += 1
        return KnowledgeBuildResult(
            backend=self.name,
            document_count=len({page.document for page in pages}),
            page_count=len(pages),
            chunk_count=len(chunks),
            index_path=str(self.index_path),
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        chunks = self._load_chunks()
        query_terms = set(query.lower().split())
        ranked = sorted(
            chunks,
            key=lambda chunk: bool(query_terms & set(chunk.content.lower().split())),
            reverse=True,
        )
        return [
            RetrievalResult(
                rank=index,
                score=(
                    0.9
                    if query_terms & set(chunk.content.lower().split())
                    else 0.1
                ),
                chunk=chunk,
            )
            for index, chunk in enumerate(ranked[:top_k], start=1)
        ]

    def health_check(self) -> bool:
        try:
            self._load_chunks()
        except Exception:
            return False
        return True

    def get_catalog(self):
        return build_knowledge_catalog(self.config.knowledge_base_path)

    def _load_chunks(self) -> list[KnowledgeChunk]:
        if self._chunks is None:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._chunks = [KnowledgeChunk.from_dict(item) for item in payload]
        return self._chunks


class FakeBackendFactory:
    def __init__(self) -> None:
        self.rebuilds = 0

    def __call__(self, config: Configuration) -> FakePersistedBackend:
        return FakePersistedBackend(config, self)


def _service(tmp_path: Path, factory: FakeBackendFactory | None = None):
    factory = factory or FakeBackendFactory()
    config = Configuration(
        document_sets_root=str(tmp_path / "document_sets"),
        knowledge_chunk_size=200,
        knowledge_chunk_overlap=20,
        enable_notes=False,
    )
    return DocumentSetService(config, backend_factory=factory), config, factory


@pytest.mark.parametrize(
    ("unsafe", "expected"),
    [
        ("../../secret.txt", "secret.txt"),
        (r"C:\Users\x\file.pdf", "file.pdf"),
        ("/foo/bar.md", "bar.md"),
        ("CON.txt", "_CON.txt"),
    ],
)
def test_upload_filename_is_sanitized(unsafe: str, expected: str) -> None:
    assert sanitize_filename(unsafe) == expected


@pytest.mark.parametrize("extension", ["exe", "py", "zip", "docx"])
def test_extension_validation_rejects_unsupported_file(
    tmp_path: Path,
    extension: str,
) -> None:
    service, _, _ = _service(tmp_path)
    record = service.create()

    with pytest.raises(DocumentValidationError, match="extension"):
        service.add_files(
            record.document_set_id,
            [DocumentUpload(f"payload.{extension}", b"binary")],
        )


def test_empty_document_does_not_build_index(tmp_path: Path) -> None:
    service, _, factory = _service(tmp_path)
    record = service.create()

    with pytest.raises(DocumentValidationError):
        service.add_files(record.document_set_id, [DocumentUpload("empty.md", b"")])

    assert factory.rebuilds == 0


def test_corrupt_file_isolated_when_valid_document_exists(tmp_path: Path) -> None:
    service, _, factory = _service(tmp_path)
    record = service.create()

    ready = service.add_files(
        record.document_set_id,
        [
            DocumentUpload("good.md", b"# Robotics\nrobotics evidence"),
            DocumentUpload("bad.pdf", b"not a pdf"),
        ],
    )

    assert ready.status.value == "ready"
    assert ready.documents == 1
    assert factory.rebuilds == 1
    assert {item.status for item in ready.files} == {"accepted", "rejected"}
    assert ready.notices


def test_document_sets_remain_isolated_under_concurrent_retrieval(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    set_a = service.create()
    set_b = service.create()
    service.add_files(
        set_a.document_set_id,
        [DocumentUpload("react.md", b"# ReAct\nreact reasoning acting")],
    )
    service.add_files(
        set_b.document_set_id,
        [DocumentUpload("robotics.md", b"# Robotics\nrobotics market report")],
    )

    def retrieve(document_set_id: str, query: str) -> set[str]:
        results, _, _ = service.get_knowledge_service(
            document_set_id
        ).retrieve_with_backend(query)
        return {item.chunk.document for item in results}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(retrieve, set_a.document_set_id, "react")
        future_b = executor.submit(retrieve, set_b.document_set_id, "robotics")

    assert future_a.result() == {"react.md"}
    assert future_b.result() == {"robotics.md"}


def test_persisted_index_loads_without_rebuild_after_service_restart(
    tmp_path: Path,
) -> None:
    service, config, factory = _service(tmp_path)
    record = service.create()
    service.add_files(
        record.document_set_id,
        [DocumentUpload("notes.md", b"# Session\npersistent session evidence")],
    )
    assert factory.rebuilds == 1

    restarted = DocumentSetService(config, backend_factory=factory)
    results, _, _ = restarted.get_knowledge_service(
        record.document_set_id
    ).retrieve_with_backend("persistent")

    assert factory.rebuilds == 1
    assert results[0].chunk.document == "notes.md"


def test_document_set_api_creates_uploads_and_reports_ready(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    with TestClient(main_module.create_app(service)) as client:
        created = client.post("/knowledge/document-sets").json()
        response = client.post(
            f"/knowledge/document-sets/{created['document_set_id']}/files",
            files=[
                (
                    "files",
                    ("../../notes.md", b"# Notes\nsession evidence", "text/markdown"),
                )
            ],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["files"][0]["name"] == "notes.md"
    assert payload["documents"] == 1


def test_document_research_requires_ready_document_set(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    with TestClient(main_module.create_app(service)) as client:
        missing = client.post(
            "/research",
            json={"topic": "test", "research_mode": "document"},
        )
        created = client.post("/knowledge/document-sets").json()
        not_ready = client.post(
            "/research",
            json={
                "topic": "test",
                "research_mode": "hybrid",
                "document_set_id": created["document_set_id"],
            },
        )

    assert missing.status_code == 400
    assert not_ready.status_code == 409
