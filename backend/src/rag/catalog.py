"""Lightweight knowledge catalog construction without embedding or retrieval."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .base import KnowledgeDocumentInfo
from .loader import SUPPORTED_EXTENSIONS


def build_knowledge_catalog(path: str | Path) -> list[KnowledgeDocumentInfo]:
    """Return truthful file/page metadata without loading document content."""

    source = Path(path).resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = [
            item
            for item in sorted(source.rglob("*"))
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        raise FileNotFoundError(f"Knowledge base path not found: {source}")

    catalog: list[KnowledgeDocumentInfo] = []
    for item in paths:
        suffix = item.suffix.lower()
        pages = len(PdfReader(str(item)).pages) if suffix == ".pdf" else 1
        catalog.append(
            KnowledgeDocumentInfo(
                document=item.name,
                file_type=suffix.lstrip("."),
                pages=pages,
            )
        )
    return catalog
