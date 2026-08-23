"""Server log credential redaction tests."""

from services.log_redaction import redact_sensitive_text


def test_log_redaction_removes_common_credentials() -> None:
    raw = (
        "Bearer bearer-value api_key=api-value secret-token=secret-value "
        "authorization: auth-value https://user:password@example.com/path"
    )

    redacted = redact_sensitive_text(raw)

    for secret in (
        "bearer-value",
        "api-value",
        "secret-value",
        "auth-value",
        "user",
        "password",
    ):
        assert secret not in redacted
    assert redacted.count("[REDACTED]") >= 5
