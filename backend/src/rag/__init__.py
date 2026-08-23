"""Local knowledge retrieval components."""

from .chunker import chunk_documents
from .embedding import HashingEmbedding
from .loader import load_document, load_documents
from .retriever import KnowledgeRetriever
from .types import DocumentPage, KnowledgeChunk, RetrievalResult
from .vector_store import FaissVectorStore

__all__ = [
    "DocumentPage",
    "KnowledgeChunk",
    "RetrievalResult",
    "FaissVectorStore",
    "HashingEmbedding",
    "KnowledgeRetriever",
    "chunk_documents",
    "load_document",
    "load_documents",
]
