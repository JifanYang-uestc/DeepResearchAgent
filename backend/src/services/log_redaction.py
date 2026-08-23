"""Redact credentials before exception details reach server logs."""

from __future__ import annotations

import re
from typing import Any

_REDACTION_PATTERNS = (
    re.compile(r"(?i)(https?://)[^/\s:@]+:[^@\s/]+@"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)\b(api[_ -]?key|secret[-_ ]?token|authorization)\b"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\btvly-[A-Za-z0-9_-]{8,}"),
)


def redact_sensitive_text(value: Any) -> str:
    """Return diagnostic text with common credential forms removed."""

    text = str(value)
    for index, pattern in enumerate(_REDACTION_PATTERNS):
        replacement = r"\1[REDACTED]@" if index == 0 else "[REDACTED]"
        text = pattern.sub(replacement, text)
    return text
