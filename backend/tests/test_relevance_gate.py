"""Knowledge relevance gate calibration and backend-specific behavior."""

from __future__ import annotations

from config import Configuration
from rag.types import KnowledgeChunk, RetrievalResult
from services.relevance_gate import KnowledgeRelevanceGate


def _result(score: float, content: str, document: str = "react.pdf") -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=score,
        chunk=KnowledgeChunk(
            chunk_id=f"chunk-{score}",
            content=content,
            document=document,
            source_path=document,
            file_type="pdf",
            page=2,
            start_char=0,
            end_char=len(content),
        ),
    )


def test_semantic_gate_rejects_weak_robotics_candidate() -> None:
    gate = KnowledgeRelevanceGate(
        Configuration(knowledge_relevance_threshold=0.55)
    )

    result = gate.filter(
        "机器人市场趋势",
        [_result(0.39, "ReAct reasoning and acting")],
        backend_name="helloagents",
    )

    assert result.accepted == []
    assert len(result.rejected) == 1
    assert result.mode == "semantic_threshold"


def test_semantic_gate_accepts_in_domain_react_candidate() -> None:
    gate = KnowledgeRelevanceGate(
        Configuration(knowledge_relevance_threshold=0.55)
    )

    result = gate.filter(
        "ReAct reasoning and acting 原理",
        [_result(0.74, "ReAct interleaves reasoning traces and actions")],
        backend_name="helloagents",
    )

    assert len(result.accepted) == 1
    assert result.rejected == []


def test_gate_keeps_top_k_distinct_from_relevant_k() -> None:
    gate = KnowledgeRelevanceGate(Configuration(knowledge_relevance_threshold=0.55))
    result = gate.filter(
        "ReAct 原理",
        [
            _result(0.72, "ReAct reasoning and acting"),
            _result(0.42, "Weakly related text", "rag_2020.pdf"),
        ],
        backend_name="helloagents",
    )

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert result.accepted[0].rank == 1


def test_legacy_gate_does_not_apply_semantic_threshold() -> None:
    gate = KnowledgeRelevanceGate(Configuration(knowledge_relevance_threshold=0.99))
    result = gate.filter(
        "ReAct reasoning and acting",
        [_result(0.03, "ReAct interleaves reasoning and acting")],
        backend_name="legacy_faiss",
    )

    assert len(result.accepted) == 1
    assert result.threshold is None
    assert result.mode == "legacy_lexical_judge"
