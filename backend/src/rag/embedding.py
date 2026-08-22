"""Dependency-light deterministic embeddings for reproducible V1 retrieval."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

import numpy as np

_LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_CJK_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class HashingEmbedding:
    """Map multilingual lexical features to normalized dense vectors.

    This local embedding keeps indexing deterministic and offline. It is designed
    as the V1 default for a small, human-verifiable corpus; the vector-store API
    intentionally keeps the embedding implementation replaceable.
    """

    name = "hashing-multilingual-v1"

    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions < 128:
            raise ValueError("Embedding dimensions must be at least 128")
        self.dimensions = dimensions

    @property
    def fingerprint(self) -> str:
        """Return a persisted identity used to reject incompatible indexes."""

        return f"{self.name}:{self.dimensions}"

    def embed(self, text: str) -> np.ndarray:
        """Embed one string as a unit-length float32 vector."""

        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature, weight in _features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a sequence into a contiguous FAISS-compatible matrix."""

        rows = [self.embed(text) for text in texts]
        if not rows:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def _features(text: str) -> list[tuple[str, float]]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    features: list[tuple[str, float]] = []

    latin_tokens = _LATIN_TOKEN.findall(normalized)
    for token in latin_tokens:
        features.append((f"w:{token}", 2.0))
        canonical = re.sub(r"[-_.]+", "-", token)
        features.append((f"wc:{canonical}", 2.0))
        features.extend(
            (f"wp:{part}", 1.0)
            for part in re.split(r"[-_.]+", token)
            if part
        )
    features.extend(
        (f"wb:{left}|{right}", 1.2)
        for left, right in zip(latin_tokens, latin_tokens[1:])
    )

    cjk = "".join(_CJK_CHAR.findall(normalized))
    features.extend((f"c:{char}", 0.4) for char in cjk)
    for size, weight in ((2, 1.0), (3, 1.5), (4, 1.8)):
        features.extend(
            (f"c{size}:{cjk[index:index + size]}", weight)
            for index in range(max(len(cjk) - size + 1, 0))
        )
    return features
