"""Exercise HelloAgents 0.2.9 semantic RAG APIs without external services.

This compatibility spike intentionally supplies a tiny in-memory vector store
to the lower-level HelloAgents pipeline. It verifies document ingestion,
LocalTransformerEmbedding, indexing, semantic search, and the structured result
shape without requiring Tavily, an LLM, or a Qdrant server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class InMemorySpikeStore:
    """Minimal vector-store contract consumed by HelloAgents RAG pipeline."""

    def __init__(self) -> None:
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.metadata: list[dict[str, Any]] = []
        self.ids: list[str] = []

    def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str],
    ) -> bool:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = list(metadata)
        self.ids = list(ids)
        return True

    def search_similar(
        self,
        query_vector: list[float],
        limit: int,
        score_threshold: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        results: list[dict[str, Any]] = []
        for index, (vector, metadata) in enumerate(zip(self.vectors, self.metadata)):
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            denominator = float(np.linalg.norm(vector)) * query_norm
            score = float(np.dot(vector, query) / denominator) if denominator else 0.0
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                {"id": self.ids[index], "score": score, "metadata": metadata}
            )
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()

    os.environ["EMBED_MODEL_TYPE"] = "local"
    os.environ["EMBED_MODEL_NAME"] = args.model

    from hello_agents.memory.embedding import refresh_embedder
    from hello_agents.memory.rag.pipeline import (
        index_chunks,
        load_and_chunk_texts,
        search_vectors,
    )
    from hello_agents.tools import RAGTool

    embedder = refresh_embedder()
    facts_path = BACKEND_DIR / "knowledge_base" / "test_facts.txt"
    chunks = load_and_chunk_texts(
        [str(facts_path)],
        chunk_size=220,
        chunk_overlap=40,
        namespace="v03_spike",
    )
    store = InMemorySpikeStore()
    index_chunks(store=store, chunks=chunks, rag_namespace="v03_spike")
    results = search_vectors(
        store=store,
        query="ResearchX-2026 使用什么 Vector Store？",
        top_k=1,
        rag_namespace="v03_spike",
    )
    if not results:
        raise RuntimeError("HelloAgents semantic retrieval returned no results")

    top = results[0]
    metadata = top.get("metadata", {})
    if "FAISS" not in str(metadata.get("content", "")):
        raise RuntimeError("Semantic retrieval did not return the expected fact")

    output = {
        "rag_tool_importable": RAGTool is not None,
        "embedding_class": type(embedder).__name__,
        "embedding_backend": getattr(embedder, "_backend", "unknown"),
        "embedding_model": args.model,
        "embedding_dimension": embedder.dimension,
        "chunks_indexed": len(chunks),
        "result_keys": sorted(top.keys()),
        "metadata_keys": sorted(metadata.keys()),
        "top_result": {
            "document": Path(str(metadata.get("source_path", ""))).name,
            "score": top.get("score"),
            "content": metadata.get("content"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("memory_id"),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
