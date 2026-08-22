"""Document loading and chunking tests."""

from __future__ import annotations

from pathlib import Path

from rag.chunker import chunk_documents
from rag.loader import load_document, load_documents


KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "knowledge_base"


def test_load_test_facts_with_metadata() -> None:
    pages = load_document(KNOWLEDGE_BASE / "test_facts.txt")

    assert len(pages) == 1
    assert pages[0].document == "test_facts.txt"
    assert pages[0].file_type == "txt"
    assert pages[0].page is None
    assert "ResearchX-2026" in pages[0].content


def test_chunking_is_deterministic_and_traceable() -> None:
    pages = load_documents(KNOWLEDGE_BASE)
    first = chunk_documents(pages, chunk_size=220, chunk_overlap=40)
    second = chunk_documents(pages, chunk_size=220, chunk_overlap=40)

    assert len(first) >= 2
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.document == "test_facts.txt" for chunk in first)
    assert all(chunk.end_char > chunk.start_char for chunk in first)
    assert any("Top-K 设置为 5" in chunk.content for chunk in first)


def test_unsupported_document_is_rejected(tmp_path: Path) -> None:
    unsupported = tmp_path / "facts.csv"
    unsupported.write_text("a,b", encoding="utf-8")

    try:
        load_document(unsupported)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Unsupported file type should raise ValueError")
