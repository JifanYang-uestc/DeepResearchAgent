"""Run one real V0.3 SSE acceptance flow and print a compact audit summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(BACKEND_DIR / ".env")

from agent import DeepResearchAgent
from config import Configuration

DEFAULT_TOPIC = (
    "从传统 RAG 到 Agentic RAG：对比 RAG、ReAct 和 Self-RAG，"
    "并结合 2026 年最新互联网资料分析发展趋势。"
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC)
    args = parser.parse_args()

    agent = DeepResearchAgent(Configuration.from_env())
    audit: dict[str, object] = {
        "topic": args.topic,
        "routes": [],
        "knowledge_rejected": [],
        "tasks": [],
        "source_types": set(),
        "done": False,
        "report_length": 0,
        "report_has_knowledge": False,
        "report_has_web": False,
    }
    for event in agent.run_stream(args.topic):
        event_type = event.get("type")
        if event_type == "retrieval_route":
            audit["routes"].append(
                {
                    "task_id": event.get("task_id"),
                    "route": event.get("route"),
                    "reason": event.get("reason"),
                    "confidence": event.get("confidence"),
                    "router_latency_ms": event.get("router_latency_ms"),
                }
            )
        elif event_type == "knowledge_rejected":
            audit["knowledge_rejected"].append(
                {
                    "task_id": event.get("task_id"),
                    "reason": event.get("reason"),
                }
            )
        elif event_type == "sources":
            for source in event.get("sources", []):
                if isinstance(source, dict) and source.get("type"):
                    audit["source_types"].add(source["type"])
        elif event_type == "task_status" and event.get("status") in {
            "completed",
            "skipped",
            "failed",
        }:
            audit["tasks"].append(
                {
                    "task_id": event.get("task_id"),
                    "status": event.get("status"),
                    "route": event.get("retrieval_route"),
                    "metrics_ms": event.get("retrieval_metrics_ms"),
                }
            )
        elif event_type == "final_report":
            report = str(event.get("report") or "")
            audit["report_length"] = len(report)
            audit["report_has_knowledge"] = "[Knowledge]" in report
            audit["report_has_web"] = "[Web]" in report
        elif event_type == "done":
            audit["done"] = True

    audit["source_types"] = sorted(audit["source_types"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
