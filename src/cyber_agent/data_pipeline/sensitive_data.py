"""Secret and clear personal-identifier detection with value-free findings."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensitiveFinding:
    kind: str
    start: int
    end: int


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "secret_assignment",
        re.compile(
            r"(?im)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|pwd|cookie|ssh_password)\b\s*[:=]\s*['\"]?[^\s'\";,]{6,}"
        ),
    ),
    ("cloud_private_key_id", re.compile(r'(?i)"private_key_id"\s*:\s*"[A-Za-z0-9]{8,}"')),
    ("email_address", re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.IGNORECASE)),
    ("phone_number", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")),
    ("us_social_security_number", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("authentication_cookie", re.compile(r"(?im)\b(?:set-cookie|cookie)\s*:\s*[^\n]{8,}")),
)


def detect_sensitive_data(text: str) -> list[SensitiveFinding]:
    findings: list[SensitiveFinding] = []
    for kind, pattern in PATTERNS:
        findings.extend(SensitiveFinding(kind, match.start(), match.end()) for match in pattern.finditer(text))
    return sorted(findings, key=lambda finding: (finding.start, finding.end, finding.kind))


def redact_sensitive_data(text: str, findings: list[SensitiveFinding]) -> str:
    if not findings:
        return text
    merged: list[tuple[int, int, set[str]]] = []
    for finding in findings:
        if merged and finding.start <= merged[-1][1]:
            start, end, kinds = merged[-1]
            merged[-1] = (start, max(end, finding.end), kinds | {finding.kind})
        else:
            merged.append((finding.start, finding.end, {finding.kind}))
    output: list[str] = []
    cursor = 0
    for start, end, kinds in merged:
        output.append(text[cursor:start])
        output.append(f"[REDACTED:{'+'.join(sorted(kinds))}]")
        cursor = end
    output.append(text[cursor:])
    return "".join(output)

