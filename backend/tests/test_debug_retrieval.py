"""Debug retrieval must never overwrite either runtime index."""

from __future__ import annotations

from scripts.debug_retrieval import BACKEND_DIR, DEFAULT_INDEX_DIR


def test_debug_retrieval_default_path_is_isolated() -> None:
    runtime_legacy = (BACKEND_DIR / "vector_store").resolve()
    runtime_semantic = (runtime_legacy / "helloagents-semantic").resolve()
    debug_index = DEFAULT_INDEX_DIR.resolve()

    assert debug_index == runtime_legacy / "debug-legacy"
    assert debug_index != runtime_legacy
    assert debug_index != runtime_semantic
