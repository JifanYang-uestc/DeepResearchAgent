"""HelloAgents semantic embedding adapter and persistence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import Configuration
from rag.helloagents_backend import (
    HelloAgentsLocalEmbedding,
    HelloAgentsSemanticBackend,
)


class FakeSemanticEmbedding:
    dimensions = 5
    fingerprint = "fake-semantic:5"

    def embed_many(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("react" in lowered or "reasoning" in lowered),
                    float("robot" in lowered or "机器人" in lowered),
                    float("alpha" in lowered),
                    float("beta" in lowered),
                    0.25,
                ]
            )
        vectors = np.asarray(rows, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_semantic_backend_persists_and_reloads_metadata(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "react.txt").write_text(
        "ReAct interleaves reasoning traces and actions with an environment.",
        encoding="utf-8",
    )
    config = Configuration(
        knowledge_base_path=str(knowledge_base),
        knowledge_index_path=str(tmp_path / "index"),
        knowledge_chunk_size=200,
        knowledge_chunk_overlap=20,
    )

    first = HelloAgentsSemanticBackend(config, embedding=FakeSemanticEmbedding())
    first_result = first.retrieve("ReAct reasoning and acting", 1)[0]
    second = HelloAgentsSemanticBackend(config, embedding=FakeSemanticEmbedding())
    second_result = second.retrieve("ReAct reasoning and acting", 1)[0]

    assert first_result.chunk.document == "react.txt"
    assert second_result.chunk.chunk_id == first_result.chunk.chunk_id
    assert first.get_catalog()[0].document == "react.txt"
    assert (tmp_path / "index" / "helloagents-semantic" / "knowledge.faiss").is_file()


def test_helloagents_embedding_normalizes_vectors(monkeypatch) -> None:
    class FakeLocalTransformerEmbedding:
        dimension = 2

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(
        "hello_agents.memory.embedding.LocalTransformerEmbedding",
        FakeLocalTransformerEmbedding,
    )
    embedding = HelloAgentsLocalEmbedding("verified-model")

    vectors = embedding.embed_many(["one", "two"])

    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert embedding.fingerprint == "helloagents-local-transformer:verified-model:2"


def _semantic_backend(tmp_path: Path) -> tuple[HelloAgentsSemanticBackend, Path]:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    config = Configuration(
        knowledge_base_path=str(knowledge_base),
        knowledge_index_path=str(tmp_path / "index"),
        knowledge_chunk_size=200,
        knowledge_chunk_overlap=20,
    )
    return (
        HelloAgentsSemanticBackend(config, embedding=FakeSemanticEmbedding()),
        knowledge_base,
    )


def test_semantic_rebuild_adds_new_document(tmp_path: Path) -> None:
    backend, knowledge_base = _semantic_backend(tmp_path)
    (knowledge_base / "a.txt").write_text("Alpha foundation facts.", encoding="utf-8")
    first = backend.rebuild()
    (knowledge_base / "b.txt").write_text("Beta newly added facts.", encoding="utf-8")

    second = backend.rebuild()

    assert first.document_count == 1
    assert second.document_count == 2
    assert backend.retrieve("Beta facts", 1)[0].chunk.document == "b.txt"


def test_semantic_rebuild_removes_deleted_document(tmp_path: Path) -> None:
    backend, knowledge_base = _semantic_backend(tmp_path)
    (knowledge_base / "a.txt").write_text("Alpha foundation facts.", encoding="utf-8")
    deleted = knowledge_base / "b.txt"
    deleted.write_text("Beta removable facts.", encoding="utf-8")
    backend.rebuild()
    deleted.unlink()

    result = backend.rebuild()
    documents = {chunk.document for chunk in backend._store.chunks}  # type: ignore[union-attr]
    retrieval_documents = {
        item.chunk.document for item in backend.retrieve("Beta facts", 5)
    }
    metadata = (Path(result.index_path) / "metadata.json").read_text(encoding="utf-8")

    assert result.document_count == 1
    assert "b.txt" not in documents
    assert "b.txt" not in retrieval_documents
    assert "b.txt" not in metadata


def test_rebuild_refreshes_cached_retriever(tmp_path: Path) -> None:
    backend, knowledge_base = _semantic_backend(tmp_path)
    (knowledge_base / "a.txt").write_text("Alpha foundation facts.", encoding="utf-8")
    backend.rebuild()
    cached_retriever = backend._get_retriever()
    backend.retrieve("Alpha", 1)
    (knowledge_base / "b.txt").write_text("Beta refreshed facts.", encoding="utf-8")

    backend.rebuild()

    assert backend._get_retriever() is not cached_retriever
    assert backend.retrieve("Beta", 1)[0].chunk.document == "b.txt"
