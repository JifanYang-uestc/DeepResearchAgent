"""Truthful catalog title extraction for adaptive routing."""

from __future__ import annotations

from pathlib import Path

from rag import catalog as catalog_module
from rag.catalog import build_knowledge_catalog


def test_catalog_extracts_markdown_and_text_titles(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "preface\n# Semantic Retrieval Notes\nbody", encoding="utf-8"
    )
    (tmp_path / "facts.txt").write_text(
        "Humanoid Robotics Field Notes\nsecond line", encoding="utf-8"
    )

    catalog = {item.document: item for item in build_knowledge_catalog(tmp_path)}

    assert catalog["notes.md"].title == "Semantic Retrieval Notes"
    assert catalog["facts.txt"].title == "Humanoid Robotics Field Notes"
    assert catalog["notes.md"].to_dict()["title"] == "Semantic Retrieval Notes"


def test_catalog_extracts_pdf_title_from_first_page(monkeypatch, tmp_path: Path) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "arXiv:2601.00001\n2026 Humanoid Robotics Industry Report\nAuthors"

    class FakeReader:
        def __init__(self, path: str) -> None:
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setattr(catalog_module, "PdfReader", FakeReader)
    (tmp_path / "document1.pdf").write_bytes(b"fake")

    entry = build_knowledge_catalog(tmp_path)[0]

    assert entry.document == "document1.pdf"
    assert entry.pages == 2
    assert entry.title == "2026 Humanoid Robotics Industry Report"


def test_corrupt_pdf_does_not_hide_valid_catalog_documents(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Valid Notes\nbody", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")

    catalog = build_knowledge_catalog(tmp_path)

    assert [entry.document for entry in catalog] == ["notes.md"]
