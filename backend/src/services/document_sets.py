"""Isolated uploaded-document lifecycle and semantic index management."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID, uuid4

from config import Configuration
from rag.base import KnowledgeBackend
from rag.helloagents_backend import HelloAgentsSemanticBackend
from rag.legacy_faiss_backend import resolve_backend_path
from rag.loader import SUPPORTED_EXTENSIONS, load_document
from services.knowledge import KnowledgeService
from services.log_redaction import redact_sensitive_text

logger = logging.getLogger(__name__)

DOCUMENT_SET_METADATA = "document_set.json"


class DocumentSetStatus(str, Enum):
    """Persisted ingestion states visible to clients."""

    UPLOADED = "uploaded"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DocumentSetError(RuntimeError):
    """Base class for safe document-set failures."""


class DocumentSetNotFound(DocumentSetError):
    """The requested UUID does not identify a document set."""


class DocumentSetNotReady(DocumentSetError):
    """The requested document set cannot serve retrieval yet."""


class DocumentValidationError(DocumentSetError):
    """Uploaded content violates a documented validation rule."""


class DocumentIndexingError(DocumentSetError):
    """Semantic ingestion failed without exposing provider details."""


@dataclass(frozen=True, slots=True)
class DocumentUpload:
    """Framework-independent uploaded file payload."""

    filename: str
    content: bytes


@dataclass(slots=True)
class DocumentFileRecord:
    """Safe per-file ingestion result."""

    name: str
    status: str
    size: int
    pages: int = 0
    notice: str | None = None


@dataclass(slots=True)
class DocumentSetRecord:
    """Persisted session knowledge scope and index statistics."""

    document_set_id: str
    status: DocumentSetStatus
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    files: list[DocumentFileRecord] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable API and persistence representation."""

        return {
            "document_set_id": self.document_set_id,
            "status": self.status.value,
            "documents": self.documents,
            "pages": self.pages,
            "chunks": self.chunks,
            "files": [asdict(item) for item in self.files],
            "notices": list(self.notices),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentSetRecord:
        """Restore a validated record from local metadata."""

        return cls(
            document_set_id=str(payload["document_set_id"]),
            status=DocumentSetStatus(str(payload["status"])),
            documents=int(payload.get("documents", 0)),
            pages=int(payload.get("pages", 0)),
            chunks=int(payload.get("chunks", 0)),
            files=[DocumentFileRecord(**item) for item in payload.get("files", [])],
            notices=[str(item) for item in payload.get("notices", [])],
        )


BackendFactory = Callable[[Configuration], KnowledgeBackend]


class DocumentSetService:
    """Create isolated corpora, build indexes, and lazily cache retrievers."""

    def __init__(
        self,
        config: Configuration,
        *,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self._config = config
        self._root = resolve_backend_path(config.document_sets_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._backend_factory = backend_factory or HelloAgentsSemanticBackend
        self._lock = RLock()
        self._cache: OrderedDict[str, KnowledgeService] = OrderedDict()

    def create(self) -> DocumentSetRecord:
        """Create one UUID-backed empty document scope."""

        with self._lock:
            document_set_id = str(uuid4())
            root = self._set_root(document_set_id)
            (root / "files").mkdir(parents=True)
            (root / "index").mkdir(parents=True)
            record = DocumentSetRecord(
                document_set_id=document_set_id,
                status=DocumentSetStatus.UPLOADED,
            )
            self._save_record(record)
            return record

    def get(self, document_set_id: str) -> DocumentSetRecord:
        """Load a document set without initializing embeddings or FAISS."""

        canonical_id = _canonical_document_set_id(document_set_id)
        metadata = self._set_root(canonical_id) / DOCUMENT_SET_METADATA
        if not metadata.is_file():
            raise DocumentSetNotFound("Document set not found")
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            record = DocumentSetRecord.from_dict(payload)
        except Exception as exc:  # noqa: BLE001 - local corruption boundary
            logger.error(
                "Document set metadata is unreadable: %s",
                redact_sensitive_text(exc),
            )
            raise DocumentSetNotReady("Document set metadata is invalid") from None
        if record.document_set_id != canonical_id:
            raise DocumentSetNotReady("Document set metadata identity mismatch")
        return record

    def add_files(
        self,
        document_set_id: str,
        uploads: Sequence[DocumentUpload],
    ) -> DocumentSetRecord:
        """Validate, save, parse, and index uploaded documents synchronously."""

        if not uploads:
            raise DocumentValidationError("At least one file is required")

        with self._lock:
            record = self.get(document_set_id)
            files_dir = self._set_root(record.document_set_id) / "files"
            existing_files = [item for item in files_dir.iterdir() if item.is_file()]
            if len(existing_files) + len(uploads) > self._config.max_upload_files:
                raise DocumentValidationError("Document set file-count limit exceeded")

            total_size = sum(item.stat().st_size for item in existing_files)
            total_size += sum(len(upload.content) for upload in uploads)
            if total_size > self._config.max_upload_total_size:
                raise DocumentValidationError("Document set total-size limit exceeded")

            for upload in uploads:
                _validate_upload(upload, self._config)

            record.status = DocumentSetStatus.INDEXING
            record.notices = []
            self._save_record(record)

            known_names = {item.name.lower() for item in existing_files}
            new_records: list[DocumentFileRecord] = []
            for upload in uploads:
                safe_name = _unique_filename(
                    sanitize_filename(upload.filename),
                    known_names,
                )
                known_names.add(safe_name.lower())
                target = files_dir / safe_name
                target.write_bytes(upload.content)
                try:
                    pages = load_document(target)
                    if not pages or not any(page.content.strip() for page in pages):
                        raise ValueError("document contains no extractable text")
                    new_records.append(
                        DocumentFileRecord(
                            name=safe_name,
                            status="accepted",
                            size=len(upload.content),
                            pages=len(pages),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - per-file isolation
                    logger.warning(
                        "Skipping unreadable uploaded document %s: %s",
                        safe_name,
                        redact_sensitive_text(exc),
                    )
                    target.unlink(missing_ok=True)
                    notice = f"{safe_name} 无法解析或没有可索引文本，已跳过。"
                    record.notices.append(notice)
                    new_records.append(
                        DocumentFileRecord(
                            name=safe_name,
                            status="rejected",
                            size=len(upload.content),
                            notice=notice,
                        )
                    )

            record.files.extend(new_records)
            if not any(files_dir.iterdir()):
                record.status = DocumentSetStatus.FAILED
                self._save_record(record)
                raise DocumentValidationError(
                    "Uploaded files contain no indexable document text"
                )

            session_config = self._session_config(record.document_set_id)
            try:
                result = self._backend_factory(session_config).rebuild()
            except Exception as exc:  # noqa: BLE001 - semantic boundary
                record.status = DocumentSetStatus.FAILED
                record.notices.append("文档索引构建失败，请检查服务端日志。")
                self._save_record(record)
                logger.error(
                    "Document set semantic indexing failed: %s",
                    redact_sensitive_text(exc),
                )
                raise DocumentIndexingError("Document index build failed") from None

            record.status = DocumentSetStatus.READY
            record.documents = result.document_count
            record.pages = result.page_count
            record.chunks = result.chunk_count
            self._cache.pop(record.document_set_id, None)
            self._save_record(record)
            return record

    def require_ready(self, document_set_id: str) -> DocumentSetRecord:
        """Return a record only when its persisted index is ready."""

        record = self.get(document_set_id)
        if record.status is not DocumentSetStatus.READY:
            raise DocumentSetNotReady("Document set is not ready")
        return record

    def get_knowledge_service(self, document_set_id: str) -> KnowledgeService:
        """Return a bounded, scope-specific retriever without Legacy fallback."""

        canonical_id = self.require_ready(document_set_id).document_set_id
        with self._lock:
            cached = self._cache.pop(canonical_id, None)
            if cached is not None:
                self._cache[canonical_id] = cached
                return cached

            session_config = self._session_config(canonical_id)
            service = KnowledgeService(
                session_config,
                backend=self._backend_factory(session_config),
                fallback_backend=None,
            )
            self._cache[canonical_id] = service
            while len(self._cache) > self._config.document_index_cache_size:
                self._cache.popitem(last=False)
            return service

    def _session_config(self, document_set_id: str) -> Configuration:
        root = self._set_root(document_set_id)
        return self._config.model_copy(
            update={
                "enable_knowledge_rag": True,
                "knowledge_backend": "helloagents",
                "knowledge_base_path": str(root / "files"),
                "knowledge_index_path": str(root / "index"),
                "knowledge_auto_build": False,
            }
        )

    def _set_root(self, document_set_id: str) -> Path:
        canonical_id = _canonical_document_set_id(document_set_id)
        target = (self._root / canonical_id).resolve()
        if target.parent != self._root.resolve():
            raise DocumentSetNotFound("Invalid document set id")
        return target

    def _save_record(self, record: DocumentSetRecord) -> None:
        root = self._set_root(record.document_set_id)
        root.mkdir(parents=True, exist_ok=True)
        target = root / DOCUMENT_SET_METADATA
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)


def sanitize_filename(value: str) -> str:
    """Remove directory semantics and unsafe characters from an upload name."""

    normalized = unicodedata.normalize("NFKC", value or "").replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1].strip()
    basename = re.sub(r"[^\w. -]", "_", basename, flags=re.UNICODE)
    basename = re.sub(r"\s+", "_", basename).lstrip(".")
    if not basename:
        basename = "document.txt"
    path = Path(basename)
    stem = path.stem[:100] or "document"
    suffix = path.suffix.lower()
    windows_reserved_names = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    if stem.rstrip(". ").lower() in windows_reserved_names:
        stem = f"_{stem}"
    return f"{stem}{suffix}"


def _validate_upload(upload: DocumentUpload, config: Configuration) -> None:
    safe_name = sanitize_filename(upload.filename)
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentValidationError("Unsupported document extension")
    if not upload.content:
        raise DocumentValidationError("Uploaded file is empty")
    if len(upload.content) > config.max_upload_file_size:
        raise DocumentValidationError("Uploaded file exceeds per-file size limit")


def _unique_filename(value: str, known_lower_names: set[str]) -> str:
    if value.lower() not in known_lower_names:
        return value
    path = Path(value)
    counter = 2
    while True:
        candidate = f"{path.stem}_{counter}{path.suffix}"
        if candidate.lower() not in known_lower_names:
            return candidate
        counter += 1


def _canonical_document_set_id(value: str) -> str:
    try:
        parsed = UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        raise DocumentSetNotFound("Invalid document set id") from None
    canonical = str(parsed)
    if canonical != str(value).lower():
        raise DocumentSetNotFound("Invalid document set id")
    return canonical
