"""Configured CORS origins must be enforced by FastAPI."""

from __future__ import annotations

from fastapi.testclient import TestClient

import main as main_module
from config import Configuration
from services.document_sets import DocumentSetService


def test_main_uses_configured_cors_origins(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://allowed.example")
    service = DocumentSetService(
        Configuration(document_sets_root=str(tmp_path / "sets"))
    )

    with TestClient(main_module.create_app(service)) as client:
        response = client.options(
            "/research",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://allowed.example"
