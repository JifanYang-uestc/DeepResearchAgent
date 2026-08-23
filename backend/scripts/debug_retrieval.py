"""Build the engineering corpus index and print human-verifiable retrieval results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
DEFAULT_INDEX_DIR = BACKEND_DIR / "vector_store" / "debug-legacy"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.chunker import chunk_documents
from rag.loader import load_document
from rag.retriever import KnowledgeRetriever
from rag.vector_store import FaissVectorStore

DEFAULT_QUERIES = [
    "ResearchX-2026 的本地知识库默认 Top-K 是多少？",
    "ResearchX-2026 包含哪些研究阶段？",
    "当本地 Knowledge RAG 不可用时系统应该怎么处理？",
    "ResearchX-2026 使用什么 Vector Store？",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Optional query; defaults to all four checks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()

    facts_path = BACKEND_DIR / "knowledge_base" / "test_facts.txt"
    chunks = chunk_documents(
        load_document(facts_path),
        chunk_size=220,
        chunk_overlap=40,
    )
    store = FaissVectorStore()
    store.build(chunks)
    store.save(args.index_dir)
    retriever = KnowledgeRetriever(store, top_k=args.top_k)

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    for query in queries:
        print(f"\nQuery: {query}")
        for result in retriever.retrieve(query):
            chunk = result.chunk
            print(f"\nTop {result.rank}")
            print(f"Rank: {result.rank}")
            print(f"Score: {result.score:.6f}")
            print(f"Document: {chunk.document}")
            print(f"Page: {chunk.page if chunk.page is not None else 'N/A'}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print("Content:")
            print(chunk.content)


if __name__ == "__main__":
    main()
