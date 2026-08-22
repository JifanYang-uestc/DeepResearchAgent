"""Build the persisted local knowledge FAISS index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.vector_store import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", type=Path, default=BACKEND_DIR / "knowledge_base")
    parser.add_argument("--index-dir", type=Path, default=BACKEND_DIR / "vector_store")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    pages = load_documents(args.knowledge_base)
    chunks = chunk_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store = FaissVectorStore()
    store.build(chunks)
    store.save(args.index_dir)
    print(
        f"Indexed {store.size} chunks from {len(pages)} pages into "
        f"{args.index_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
