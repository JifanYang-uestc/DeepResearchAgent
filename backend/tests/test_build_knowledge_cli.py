"""Knowledge build CLI configuration precedence tests."""

from __future__ import annotations

from scripts.build_knowledge_index import _build_parser, configuration_from_args


def test_build_cli_respects_env_when_flags_omitted(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", "env-corpus")
    monkeypatch.setenv("KNOWLEDGE_INDEX_PATH", "env-index")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_SIZE", "640")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_OVERLAP", "64")
    monkeypatch.setenv("KNOWLEDGE_BACKEND", "legacy_faiss")

    config = configuration_from_args(_build_parser().parse_args([]))

    assert config.knowledge_base_path == "env-corpus"
    assert config.knowledge_index_path == "env-index"
    assert config.knowledge_chunk_size == 640
    assert config.knowledge_chunk_overlap == 64
    assert config.knowledge_backend == "legacy_faiss"


def test_build_cli_explicit_flags_override_env(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", "env-corpus")
    monkeypatch.setenv("KNOWLEDGE_INDEX_PATH", "env-index")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_SIZE", "640")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_OVERLAP", "64")
    monkeypatch.setenv("KNOWLEDGE_BACKEND", "legacy_faiss")

    args = _build_parser().parse_args(
        [
            "--knowledge-base",
            "cli-corpus",
            "--index-dir",
            "cli-index",
            "--chunk-size",
            "900",
            "--chunk-overlap",
            "90",
            "--backend",
            "helloagents",
        ]
    )
    config = configuration_from_args(args)

    assert config.knowledge_base_path == "cli-corpus"
    assert config.knowledge_index_path == "cli-index"
    assert config.knowledge_chunk_size == 900
    assert config.knowledge_chunk_overlap == 90
    assert config.knowledge_backend == "helloagents"
