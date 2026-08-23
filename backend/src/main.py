"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

from agent import DeepResearchAgent
from config import Configuration, SearchAPI
from research_mode import ResearchMode
from services.document_sets import (
    DocumentIndexingError,
    DocumentSetNotFound,
    DocumentSetNotReady,
    DocumentSetService,
    DocumentUpload,
    DocumentValidationError,
)
from services.knowledge import KnowledgeService
from services.log_redaction import redact_sensitive_text
from services.user_messages import (
    INVALID_RESEARCH_REQUEST,
    RESEARCH_FAILED,
    STREAMING_RESEARCH_FAILED,
)

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    research_mode: ResearchMode = Field(
        default=ResearchMode.WEB,
        description="User-controlled source permission boundary",
    )
    document_set_id: str | None = Field(
        default=None,
        description="Ready uploaded-document scope required by document/hybrid modes",
    )
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    research_mode: ResearchMode
    document_set_id: str | None = None
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)


def create_app(
    document_set_service: DocumentSetService | None = None,
) -> FastAPI:
    initial_config = Configuration.from_env()
    app = FastAPI(title="HelloAgents Deep Researcher")
    document_sets = document_set_service or DocumentSetService(initial_config)
    app.state.document_sets = document_sets

    app.add_middleware(
        CORSMiddleware,
        allow_origins=initial_config.resolved_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider={} model={} base_url={} search_api={} "
            "max_loops={} fetch_full_page={} tool_calling={} strip_thinking={} api_key={}",
            config.llm_provider,
            config.resolved_model() or "unset",
            redact_sensitive_text(base_url),
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            "configured" if config.llm_api_key else "unset",
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/knowledge/document-sets")
    def create_document_set() -> dict[str, Any]:
        return document_sets.create().to_dict()

    @app.post("/knowledge/document-sets/{document_set_id}/files")
    async def upload_document_set_files(
        document_set_id: str,
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if len(files) > initial_config.max_upload_files:
            raise HTTPException(status_code=400, detail="上传文件数量超过限制。")
        uploads: list[DocumentUpload] = []
        total_size = 0
        try:
            for item in files:
                content = await item.read(initial_config.max_upload_file_size + 1)
                if len(content) > initial_config.max_upload_file_size:
                    raise HTTPException(status_code=413, detail="单个上传文件超过大小限制。")
                total_size += len(content)
                if total_size > initial_config.max_upload_total_size:
                    raise HTTPException(status_code=413, detail="本次上传文件总大小超过限制。")
                uploads.append(
                    DocumentUpload(
                        filename=item.filename or "document.txt",
                        content=content,
                    )
                )
        finally:
            for item in files:
                await item.close()

        try:
            return document_sets.add_files(document_set_id, uploads).to_dict()
        except DocumentSetNotFound as exc:
            raise HTTPException(status_code=404, detail="文档集不存在。") from exc
        except DocumentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DocumentIndexingError as exc:
            raise HTTPException(status_code=503, detail="文档索引构建失败。") from exc

    @app.get("/knowledge/document-sets/{document_set_id}")
    def get_document_set(document_set_id: str) -> dict[str, Any]:
        try:
            return document_sets.get(document_set_id).to_dict()
        except DocumentSetNotFound as exc:
            raise HTTPException(status_code=404, detail="文档集不存在。") from exc
        except DocumentSetNotReady as exc:
            raise HTTPException(status_code=409, detail="文档集状态不可用。") from exc

    def build_agent(payload: ResearchRequest) -> DeepResearchAgent:
        config = _build_config(payload)
        knowledge: KnowledgeService | None = None
        if payload.research_mode in {ResearchMode.DOCUMENT, ResearchMode.HYBRID}:
            if not payload.document_set_id:
                raise HTTPException(
                    status_code=400,
                    detail="Document/Hybrid 模式必须提供 document_set_id。",
                )
            try:
                knowledge = document_sets.get_knowledge_service(payload.document_set_id)
            except DocumentSetNotFound as exc:
                raise HTTPException(status_code=404, detail="文档集不存在。") from exc
            except DocumentSetNotReady as exc:
                raise HTTPException(status_code=409, detail="文档集尚未就绪。") from exc

        return DeepResearchAgent(
            config=config,
            research_mode=payload.research_mode,
            document_set_id=(
                payload.document_set_id
                if payload.research_mode is not ResearchMode.WEB
                else None
            ),
            knowledge=knowledge,
        )

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            agent = build_agent(payload)
            result = agent.run(payload.topic)
        except HTTPException:
            raise
        except ValueError as exc:  # Likely due to unsupported configuration
            logger.error(
                "Invalid synchronous research request: {}",
                redact_sensitive_text(exc),
            )
            raise HTTPException(
                status_code=400,
                detail=INVALID_RESEARCH_REQUEST,
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.error("Synchronous research failed: {}", redact_sensitive_text(exc))
            raise HTTPException(status_code=500, detail=RESEARCH_FAILED) from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "sources": item.source_items,
                "note_id": item.note_id,
                "note_path": item.note_path,
                "retrieval_route": item.retrieval_route,
                "retrieval_reason": item.retrieval_reason,
                "retrieval_confidence": item.retrieval_confidence,
                "freshness_required": item.freshness_required,
                "retrieval_metrics_ms": item.retrieval_metrics_ms,
                "notices": item.notices,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            research_mode=payload.research_mode,
            document_set_id=(
                payload.document_set_id
                if payload.research_mode is not ResearchMode.WEB
                else None
            ),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        try:
            agent = build_agent(payload)
        except HTTPException:
            raise
        except ValueError as exc:
            logger.error(
                "Invalid streaming research request: {}",
                redact_sensitive_text(exc),
            )
            raise HTTPException(
                status_code=400,
                detail=INVALID_RESEARCH_REQUEST,
            ) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.error(
                    "Streaming research failed: {}",
                    redact_sensitive_text(exc),
                )
                error_payload = {
                    "type": "error",
                    "detail": STREAMING_RESEARCH_FAILED,
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
