"""Deterministic, metadata-preserving document chunking."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .types import DocumentPage, KnowledgeChunk


def chunk_documents(
    pages: Iterable[DocumentPage],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[KnowledgeChunk]:
    """Split pages into overlapping chunks near natural text boundaries."""

    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[KnowledgeChunk] = []
    for page in pages:
        text = _normalize_text(page.content)
        start = 0
        while start < len(text):
            target_end = min(start + chunk_size, len(text))
            end = _find_boundary(text, start, target_end, chunk_size)
            content = text[start:end].strip()
            if content:
                chunk_id = _chunk_id(page, start, end, content)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        content=content,
                        document=page.document,
                        source_path=page.source_path,
                        file_type=page.file_type,
                        page=page.page,
                        start_char=start,
                        end_char=end,
                    )
                )
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_boundary(text: str, start: int, target_end: int, chunk_size: int) -> int:
    if target_end >= len(text):
        return len(text)

    search_floor = start + int(chunk_size * 0.6)
    candidates = [
        text.rfind(separator, search_floor, target_end)
        for separator in ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")
    ]
    boundary = max(candidates, default=-1)
    if boundary <= start:
        return target_end
    return boundary + 1


def _chunk_id(page: DocumentPage, start: int, end: int, content: str) -> str:
    identity = f"{page.source_path}|{page.page}|{start}|{end}|{content}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:20]
