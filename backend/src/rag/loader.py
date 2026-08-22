"""Document loading for the V1 local knowledge base."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader

from .types import DocumentPage

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown"})


def load_document(path: str | Path) -> list[DocumentPage]:
    """Load a supported document while preserving source/page metadata."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Knowledge document not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported knowledge document type: {suffix}")

    if suffix == ".pdf":
        return _load_pdf(source)

    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    return [
        DocumentPage(
            content=text,
            document=source.name,
            source_path=str(source),
            file_type=suffix.lstrip("."),
        )
    ]


def load_documents(path: str | Path) -> list[DocumentPage]:
    """Load one file or every supported file below a directory."""

    source = Path(path).resolve()
    if source.is_file():
        return load_document(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Knowledge base path not found: {source}")

    documents: list[DocumentPage] = []
    for file_path in _iter_supported_files(source):
        documents.extend(load_document(file_path))
    return documents


def _iter_supported_files(directory: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_pdf(source: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    reader = PdfReader(str(source))
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(
            DocumentPage(
                content=text,
                document=source.name,
                source_path=str(source),
                file_type="pdf",
                page=page_number,
            )
        )
    return pages
