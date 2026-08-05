from __future__ import annotations

from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.extract import html_to_text, validate_utf8
from cyber_agent.data_pipeline.normalize import normalize_text
from cyber_agent.data_pipeline.quality import assess_quality
from cyber_agent.data_pipeline.sensitive_data import detect_sensitive_data, redact_sensitive_data


def test_unicode_is_normalized_to_nfc() -> None:
    decomposed = "Cafe\u0301 security guidance"
    assert normalize_text(decomposed, preserve_code=False) == "Café security guidance"


def test_html_cleanup_removes_navigation_and_scripts() -> None:
    raw = "<header>Repeated header</header><nav>Home Pricing</nav><main><h1>Guide</h1><p>Inspect the service safely.</p></main><script>secret()</script>"
    cleaned = normalize_text(html_to_text(raw), preserve_code=False)
    assert "Guide" in cleaned
    assert "Inspect the service safely." in cleaned
    assert "Home Pricing" not in cleaned
    assert "secret()" not in cleaned


def test_code_formatting_preserves_leading_whitespace() -> None:
    code = "def check(value):\r\n    if value:\r\n        return {\"safe\": True}\r\n"
    assert normalize_text(code, preserve_code=True) == (
        'def check(value):\n    if value:\n        return {"safe": True}'
    )


def test_prose_heading_adornments_are_removed_before_repetition_checks() -> None:
    prose = "Security logging\n****************\n\nKeep an auditable event timeline."
    assert normalize_text(prose, preserve_code=False) == (
        "Security logging\n\nKeep an auditable event timeline."
    )
    code = "banner = '****************'\n"
    assert normalize_text(code, preserve_code=True) == "banner = '****************'"


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("api_key = 'synthetic-secret-value'", "secret_assignment"),
        ("token AKIAIOSFODNN7EXAMPLE end", "aws_access_key"),
        ("-----BEGIN PRIVATE KEY-----", "private_key"),
        ("Contact analyst@example.org for access", "email_address"),
        ("Call 415-555-0199 for support", "phone_number"),
        ("Set-Cookie: session=synthetic-cookie", "authentication_cookie"),
    ],
)
def test_sensitive_and_personal_data_detection(text: str, expected_kind: str) -> None:
    findings = detect_sensitive_data(text)
    assert expected_kind in {finding.kind for finding in findings}
    redacted = redact_sensitive_data(text, findings)
    assert "[REDACTED:" in redacted
    assert text != redacted


def test_empty_corrupt_and_repetitive_records_are_rejected(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    with pytest.raises(ValueError, match="valid UTF-8"):
        validate_utf8(b"\xff\xfe")
    empty = assess_quality("", "general", config)
    assert not empty.accepted
    assert "empty" in empty.reason_codes
    repetitive = assess_quality("SECURE " * 100, "general", config)
    assert not repetitive.accepted
    assert "pathological_repetition" in repetitive.reason_codes
    garbage = assess_quality("brrr cwmpt zzzq trkly plmnr grrr qwrty xkcdq brxnt qqqr zzzm trkpt", "general", config)
    assert not garbage.accepted
    assert "generated_garbage" in garbage.reason_codes
