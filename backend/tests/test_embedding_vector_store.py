"""Embedding and FAISS persistence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rag.chunker import chunk_documents
from rag.embedding import HashingEmbedding
from rag.loader import load_document
from rag.vector_store import FaissVectorStore


KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "knowledge_base"


def test_hashing_embedding_is_deterministic_and_normalized() -> None:
    embedding = HashingEmbedding(dimensions=512)
    first = embedding.embed("ResearchX-2026 使用 FAISS")
    second = embedding.embed("ResearchX-2026 使用 FAISS")

    assert np.array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_faiss_index_round_trip(tmp_path: Path) -> None:
    pages = load_document(KNOWLEDGE_BASE / "test_facts.txt")
    chunks = chunk_documents(pages, chunk_size=220, chunk_overlap=40)
    index_dir = tmp_path / "vector_store"

    store = FaissVectorStore(HashingEmbedding(dimensions=512))
    store.build(chunks)
    before = store.search("ResearchX-2026 使用什么 Vector Store？", top_k=3)
    store.save(index_dir)

    restored = FaissVectorStore(HashingEmbedding(dimensions=512))
    restored.load(index_dir)
    after = restored.search("ResearchX-2026 使用什么 Vector Store？", top_k=3)

    assert restored.size == len(chunks)
    assert [result.chunk.chunk_id for result in before] == [
        result.chunk.chunk_id for result in after
    ]
    assert any("FAISS" in result.chunk.content for result in after)
