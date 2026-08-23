"""User-controlled research source permissions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResearchMode(str, Enum):
    """Authoritative source permissions for one research request."""

    WEB = "web"
    DOCUMENT = "document"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ResearchSession:
    """Request-scoped research identity and document boundary."""

    session_id: str
    topic: str
    mode: ResearchMode
    document_set_id: str | None = None
