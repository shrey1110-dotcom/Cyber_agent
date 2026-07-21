"""Strongly typed records used at every pipeline boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


Category = Literal["general", "code", "linux", "networking", "cybersecurity", "terminal"]
SplitName = Literal["train", "validation", "test"]
RejectionStage = Literal["ingest", "extract", "clean", "sensitive_data", "quality"]

CATEGORIES = frozenset({"general", "code", "linux", "networking", "cybersecurity", "terminal"})


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_document_id(source_name: str, source_url: str) -> str:
    identity = f"{source_name.strip()}\n{source_url.strip()}".encode("utf-8")
    return f"doc_{hashlib.sha256(identity).hexdigest()}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RawDocument:
    document_id: str
    raw_text: str
    source_name: str
    source_url: str
    license: str
    category: Category
    language: str
    retrieved_at: str
    media_type: str = "text/plain"
    attribution_requirements: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("document_id", "source_name", "source_url", "license", "retrieved_at"):
            _require_nonempty(getattr(self, name), name)
        if self.category not in CATEGORIES:
            raise ValueError(f"unsupported category: {self.category}")
        if self.language != "en":
            raise ValueError("Phase 2 accepts only language='en'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RawDocument:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    text: str
    source_name: str
    source_url: str
    license: str
    category: Category
    language: str
    retrieved_at: str
    content_hash: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "document_id",
            "text",
            "source_name",
            "source_url",
            "license",
            "retrieved_at",
            "content_hash",
        ):
            _require_nonempty(getattr(self, name), name)
        if self.category not in CATEGORIES:
            raise ValueError(f"unsupported category: {self.category}")
        if self.language != "en":
            raise ValueError("Phase 2 accepts only language='en'")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
        if self.content_hash != sha256_text(self.text):
            raise ValueError("content_hash does not match text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Document:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    document_id: str
    source_name: str
    stage: RejectionStage
    reason: str
    reason_codes: tuple[str, ...]
    rejected_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RejectionRecord:
        copied = dict(value)
        copied["reason_codes"] = tuple(copied.get("reason_codes", ()))
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    removed_document_id: str
    kept_document_id: str
    duplicate_type: Literal["exact", "near"]
    similarity: float
    duplicate_group_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    document_id: str
    duplicate_group_id: str
    split: SplitName
    source_name: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

