"""Lightweight knowledge catalog construction without embedding or retrieval."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

from services.log_redaction import redact_sensitive_text

from .base import KnowledgeDocumentInfo
from .loader import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


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
        try:
            suffix = item.suffix.lower()
            reader = PdfReader(str(item)) if suffix == ".pdf" else None
            pages = len(reader.pages) if reader is not None else 1
            catalog.append(
                KnowledgeDocumentInfo(
                    document=item.name,
                    file_type=suffix.lstrip("."),
                    pages=pages,
                    title=_extract_title(item, reader),
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate corrupt documents
            logger.warning(
                "Skipping unreadable catalog document %s: %s",
                item.name,
                redact_sensitive_text(exc),
            )
    return catalog


def _extract_title(path: Path, reader: PdfReader | None) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            match = re.match(r"^\s*#\s+(.+?)\s*$", line)
            if match:
                return _clean_title(match.group(1))
        return None
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        return _first_title_line(text)
    if suffix == ".pdf" and reader is not None and reader.pages:
        text = reader.pages[0].extract_text() or ""
        return _first_title_line(text, skip_document_noise=True)
    return None


def _first_title_line(text: str, *, skip_document_noise: bool = False) -> str | None:
    for line in text.splitlines():
        candidate = _clean_title(line)
        if candidate is None:
            continue
        lowered = candidate.lower()
        if skip_document_noise and (
            lowered.startswith(("arxiv:", "doi:", "http://", "https://"))
            or re.fullmatch(r"(?:page\s*)?\d+", lowered)
        ):
            continue
        return candidate
    return None


def _clean_title(value: str) -> str | None:
    candidate = " ".join(value.strip().lstrip("#").split())
    if not 3 <= len(candidate) <= 200:
        return None
    return candidate
