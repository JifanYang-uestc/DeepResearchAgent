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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", type=Path, default=BACKEND_DIR / "knowledge_base")
    parser.add_argument("--index-dir", type=Path, default=BACKEND_DIR / "vector_store")
    parser.add_argument(
        "--backend",
        choices=("helloagents", "legacy_faiss"),
        default=None,
        help="Override KNOWLEDGE_BACKEND for this build",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    config = Configuration.from_env(
        overrides={
            "knowledge_backend": args.backend,
            "knowledge_base_path": str(args.knowledge_base),
            "knowledge_index_path": str(args.index_dir),
            "knowledge_chunk_size": args.chunk_size,
            "knowledge_chunk_overlap": args.chunk_overlap,
        }
    )
    service = KnowledgeService(config)
    ready, notices = service.prepare()
    for notice in notices:
        print(notice)
    if not ready:
        raise SystemExit(1)
    catalog, catalog_notices = service.get_catalog()
    for notice in catalog_notices:
        print(notice)
    print(f"Knowledge backend ready: {service.backend_name}")
    print(f"Documents: {len(catalog)}")
    print(f"Index root: {args.index_dir.resolve()}")


if __name__ == "__main__":
    main()
