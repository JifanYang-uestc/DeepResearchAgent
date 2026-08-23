"""Opt-in real local embedding acceptance for isolated Session RAG."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from config import Configuration
from services.document_sets import DocumentSetService, DocumentUpload

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SEMANTIC_LIVE") != "1",
    reason="set RUN_SEMANTIC_LIVE=1 for local semantic acceptance",
)


def test_session_semantic_index_persists_and_reloads(tmp_path: Path) -> None:
    config = Configuration(
        document_sets_root=str(tmp_path / "document_sets"),
        knowledge_chunk_size=240,
        knowledge_chunk_overlap=30,
        knowledge_relevance_threshold=0.4,
        enable_notes=False,
    )
    service = DocumentSetService(config)
    record = service.create()
    ready = service.add_files(
        record.document_set_id,
        [
            DocumentUpload(
                "react.md",
                (
                    "# ReAct\nReAct interleaves language model reasoning traces with "
                    "actions and observations."
                ).encode(),
            )
        ],
    )

    restarted = DocumentSetService(config)
    results, notices, backend = restarted.get_knowledge_service(
        ready.document_set_id
    ).retrieve_with_backend("How does ReAct combine reasoning and acting?")

    assert ready.status.value == "ready"
    assert backend == "helloagents"
    assert notices == []
    assert results
    assert results[0].chunk.document == "react.md"
