"""Print Semantic candidates and Knowledge Relevance Gate decisions."""

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
from services.relevance_gate import KnowledgeRelevanceGate

DEFAULT_QUERIES = [
    "ReAct reasoning and acting 原理",
    "RAG parametric 与 non-parametric memory",
    "Self-RAG adaptive retrieval reflection critique",
    "机器人领域的发展趋势",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Optional query; defaults to calibration set")
    args = parser.parse_args()

    config = Configuration.from_env()
    service = KnowledgeService(config)
    gate = KnowledgeRelevanceGate(config)
    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    for query in queries:
        candidates, notices, backend = service.retrieve_with_backend(
            query,
            top_k=config.knowledge_probe_top_k,
        )
        result = gate.filter(query, candidates, backend_name=backend)
        print(f"\nQuery: {query}")
        print(f"Backend: {backend}")
        print(f"Gate: {result.reason}")
        accepted_ids = {item.chunk.chunk_id for item in result.accepted}
        for notice in notices:
            print(f"Notice: {notice}")
        for candidate in candidates:
            status = "ACCEPT" if candidate.chunk.chunk_id in accepted_ids else "REJECT"
            chunk = candidate.chunk
            print(f"\nTop {candidate.rank} [{status}]")
            print(f"Score: {candidate.score:.6f}")
            print(f"Document: {chunk.document}")
            print(f"Page: {chunk.page if chunk.page is not None else 'N/A'}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print("Content:")
            print(chunk.content)


if __name__ == "__main__":
    main()
