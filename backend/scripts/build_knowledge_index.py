"""Build the configured persistent Knowledge Backend index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Configuration
from services.knowledge import KnowledgeService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=("helloagents", "legacy_faiss"),
        default=None,
        help="Override KNOWLEDGE_BACKEND for this build",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    return parser


def configuration_from_args(args: argparse.Namespace) -> Configuration:
    """Apply only explicit CLI flags over environment configuration."""

    return Configuration.from_env(
        overrides={
            "knowledge_backend": args.backend,
            "knowledge_base_path": (
                str(args.knowledge_base) if args.knowledge_base is not None else None
            ),
            "knowledge_index_path": (
                str(args.index_dir) if args.index_dir is not None else None
            ),
            "knowledge_chunk_size": args.chunk_size,
            "knowledge_chunk_overlap": args.chunk_overlap,
        }
    )


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    config = configuration_from_args(args)
    service = KnowledgeService(config)
    result = service.rebuild()
    print(f"Knowledge backend rebuilt: {result.backend}")
    print(f"Documents: {result.document_count}")
    print(f"Pages: {result.page_count}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Index: {result.index_path}")


if __name__ == "__main__":
    main()
