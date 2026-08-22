"""Retrieval checks over the optional three-paper demo corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.retriever import KnowledgeRetriever
from rag.vector_store import FaissVectorStore


KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "knowledge_base"
REQUIRED_PAPERS = ("rag_2020.pdf", "react.pdf", "self_rag.pdf")


@pytest.fixture(scope="module")
def paper_retriever() -> KnowledgeRetriever:
    missing = [name for name in REQUIRED_PAPERS if not (KNOWLEDGE_BASE / name).is_file()]
    if missing:
        pytest.skip(f"Optional paper corpus is not installed: {', '.join(missing)}")

    pages = load_documents(KNOWLEDGE_BASE)
    chunks = chunk_documents(pages, chunk_size=1200, chunk_overlap=180)
    store = FaissVectorStore()
    store.build(chunks)
    return KnowledgeRetriever(store, top_k=3)


@pytest.mark.parametrize(
    ("query", "expected_document"),
    [
        (
            "Retrieval-Augmented Generation parametric non-parametric memory "
            "dense retrieval knowledge-intensive NLP",
            "rag_2020.pdf",
        ),
        (
            "ReAct interleaving reasoning traces and actions external environments",
            "react.pdf",
        ),
        (
            "Self-RAG reflection tokens adaptive retrieval critique self-reflection",
            "self_rag.pdf",
        ),
    ],
)
def test_each_method_retrieves_its_paper(
    paper_retriever: KnowledgeRetriever,
    query: str,
    expected_document: str,
) -> None:
    results = paper_retriever.retrieve(query)

    assert results
    assert results[0].chunk.document == expected_document


def test_cross_document_queries_cover_all_three_papers(
    paper_retriever: KnowledgeRetriever,
) -> None:
    queries = {
        "rag_2020.pdf": (
            "Retrieval-Augmented Generation parametric non-parametric memory "
            "dense retrieval knowledge-intensive NLP"
        ),
        "react.pdf": "ReAct reasoning acting tool environment interaction",
        "self_rag.pdf": "Self-RAG adaptive retrieval reflection critique tokens",
    }
    retrieved = {
        expected: paper_retriever.retrieve(query, top_k=1)[0].chunk.document
        for expected, query in queries.items()
    }

    assert retrieved == {name: name for name in queries}
