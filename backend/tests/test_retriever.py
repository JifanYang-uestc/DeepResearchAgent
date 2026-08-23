"""Engineering-corpus retrieval acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunker import chunk_documents
from rag.loader import load_document
from rag.retriever import KnowledgeRetriever
from rag.vector_store import FaissVectorStore

KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "knowledge_base"


@pytest.fixture(scope="module")
def retriever() -> KnowledgeRetriever:
    pages = load_document(KNOWLEDGE_BASE / "test_facts.txt")
    chunks = chunk_documents(pages, chunk_size=220, chunk_overlap=40)
    store = FaissVectorStore()
    store.build(chunks)
    return KnowledgeRetriever(store, top_k=5)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ResearchX-2026 的本地知识库默认 Top-K 是多少？", "Top-K 设置为 5"),
        ("ResearchX-2026 包含哪些研究阶段？", "Planner、Retriever、Summarizer 和 Writer"),
        ("当本地 Knowledge RAG 不可用时系统应该怎么处理？", "退化到 Web Search"),
        ("ResearchX-2026 使用什么 Vector Store？", "FAISS"),
    ],
)
def test_required_queries_retrieve_expected_fact(
    retriever: KnowledgeRetriever,
    query: str,
    expected: str,
) -> None:
    results = retriever.retrieve(query)

    assert results
    assert expected in results[0].chunk.content
    assert results[0].chunk.document == "test_facts.txt"
    assert results[0].chunk.chunk_id
