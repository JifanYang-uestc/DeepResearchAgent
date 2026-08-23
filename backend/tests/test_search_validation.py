"""Untrusted Web Search response normalization tests."""

from __future__ import annotations

from services.search import MAX_TEXT_CHARS, normalize_search_response


def test_non_mapping_search_response_becomes_safe_empty_payload() -> None:
    payload, response_issue = normalize_search_response(["unexpected"], "tavily")

    assert response_issue is True
    assert payload == {
        "results": [],
        "backend": "tavily",
        "answer": None,
        "notices": [],
    }


def test_search_response_rejects_embedded_credentials_and_bounds_text() -> None:
    payload, response_issue = normalize_search_response(
        {
            "results": [
                {
                    "title": "Credential leak",
                    "url": "https://user:secret@example.com/private",
                    "content": "must be rejected",
                },
                {
                    "title": "Safe",
                    "url": "https://example.com/safe",
                    "content": "x" * (MAX_TEXT_CHARS + 100),
                },
            ],
            "answer": {"unexpected": "object"},
            "notices": ["provider detail"],
        },
        "tavily",
    )

    assert response_issue is True
    assert [item["url"] for item in payload["results"]] == [
        "https://example.com/safe"
    ]
    assert len(payload["results"][0]["content"]) == MAX_TEXT_CHARS
    assert payload["answer"] is None
    assert payload["notices"] == []
