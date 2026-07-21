"""Reviewable pipeline, path, source, and license configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


LicenseStatus = Literal["allowed", "review_required", "denied"]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    project_root: Path

    @property
    def data(self) -> Path:
        return self.project_root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def extracted(self) -> Path:
        return self.data / "extracted"

    @property
    def cleaned(self) -> Path:
        return self.data / "cleaned"

    @property
    def rejected(self) -> Path:
        return self.data / "rejected"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def splits(self) -> Path:
        return self.data / "splits"

    @property
    def reports(self) -> Path:
        return self.data / "reports"

    @property
    def configuration(self) -> Path:
        return self.project_root / "config"

    def ensure_directories(self) -> None:
        for path in (
            self.raw,
            self.extracted,
            self.cleaned,
            self.rejected,
            self.manifests,
            self.splits,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class LicenseRule:
    identifier: str
    status: LicenseStatus
    attribution_required: bool


@dataclass(frozen=True, slots=True)
class LicensePolicy:
    policy_version: int
    rules: dict[str, LicenseRule]

    def rule_for(self, identifier: str) -> LicenseRule | None:
        return self.rules.get(identifier)

    def require_allowed(self, identifier: str) -> LicenseRule:
        if not identifier.strip():
            raise ValueError("missing license information")
        rule = self.rule_for(identifier)
        if rule is None:
            raise ValueError(f"unknown license: {identifier}")
        if rule.status != "allowed":
            raise ValueError(f"license is not allowed: {identifier} ({rule.status})")
        return rule


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    paths: PipelinePaths
    minimum_document_characters: int
    maximum_document_characters: int
    minimum_quality_score: float
    near_duplicate_hamming_distance: int
    split_proportions: dict[str, float]
    default_seed: int
    sensitive_data_action: Literal["reject", "redact"]
    license_policy: LicensePolicy

    @classmethod
    def load(cls, project_root: Path | None = None) -> PipelineConfig:
        paths = PipelinePaths((project_root or default_project_root()).resolve())
        settings = _load_json(paths.configuration / "data_pipeline.json")
        raw_policy = _load_json(paths.configuration / "license_policy.json")
        rules = {
            identifier: LicenseRule(identifier, value["status"], value["attribution_required"])
            for identifier, value in raw_policy["rules"].items()
        }
        proportions = settings["split_proportions"]
        if abs(sum(proportions.values()) - 1.0) > 1e-9:
            raise ValueError("split proportions must sum to 1.0")
        config = cls(
            paths=paths,
            minimum_document_characters=int(settings["minimum_document_characters"]),
            maximum_document_characters=int(settings["maximum_document_characters"]),
            minimum_quality_score=float(settings["minimum_quality_score"]),
            near_duplicate_hamming_distance=int(settings["near_duplicate_hamming_distance"]),
            split_proportions={name: float(value) for name, value in proportions.items()},
            default_seed=int(settings["default_seed"]),
            sensitive_data_action=settings["sensitive_data_action"],
            license_policy=LicensePolicy(int(raw_policy["policy_version"]), rules),
        )
        config.paths.ensure_directories()
        return config

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "minimum_document_characters": self.minimum_document_characters,
            "maximum_document_characters": self.maximum_document_characters,
            "minimum_quality_score": self.minimum_quality_score,
            "near_duplicate_hamming_distance": self.near_duplicate_hamming_distance,
            "split_proportions": self.split_proportions,
            "default_seed": self.default_seed,
            "sensitive_data_action": self.sensitive_data_action,
            "license_policy_version": self.license_policy.policy_version,
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a JSON object: {path}")
    return value

