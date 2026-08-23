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
    dimensions = 3
    fingerprint = "fake-semantic:3"

    def embed_many(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("react" in lowered or "reasoning" in lowered),
                    float("robot" in lowered or "机器人" in lowered),
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
