"""Reviewable pipeline, path, source, and license configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal


LicenseStatus = Literal["allowed", "review_required", "denied"]


@dataclass(frozen=True, slots=True)
class DatasetMode:
    dataset_mode: Literal["audited_release", "local_research_only"]
    release_cleared: bool
    weight_publication_allowed: bool
    dataset_redistribution_allowed: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetMode:
        mode = value.get("dataset_mode")
        if mode not in {"audited_release", "local_research_only"}:
            raise ValueError("dataset_mode must be audited_release or local_research_only")
        settings = cls(
            dataset_mode=mode,
            release_cleared=value.get("release_cleared"),
            weight_publication_allowed=value.get("weight_publication_allowed"),
            dataset_redistribution_allowed=value.get("dataset_redistribution_allowed"),
        )
        if any(not isinstance(flag, bool) for flag in (
            settings.release_cleared,
            settings.weight_publication_allowed,
            settings.dataset_redistribution_allowed,
        )):
            raise ValueError("dataset publication flags must be booleans")
        if settings.dataset_mode == "local_research_only" and any((
            settings.release_cleared,
            settings.weight_publication_allowed,
            settings.dataset_redistribution_allowed,
        )):
            raise ValueError("local_research_only mode cannot enable release or redistribution")
        return settings

    @property
    def local_research_only(self) -> bool:
        return self.dataset_mode == "local_research_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_mode": self.dataset_mode,
            "release_cleared": self.release_cleared,
            "weight_publication_allowed": self.weight_publication_allowed,
            "dataset_redistribution_allowed": self.dataset_redistribution_allowed,
        }


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    project_root: Path
    collection: str | None = None

    def __post_init__(self) -> None:
        if self.collection is not None and (
            not self.collection
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in self.collection
            )
        ):
            raise ValueError("collection name must use only letters, digits, dot, underscore, and hyphen")

    @property
    def data(self) -> Path:
        base = self.project_root / "data"
        return base if self.collection is None else base / "collections" / self.collection

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
    def downloads(self) -> Path:
        return self.data / "downloads"

    @property
    def generated(self) -> Path:
        return self.data / "generated"

    @property
    def sources(self) -> Path:
        return self.data / "sources"

    @property
    def snapshots(self) -> Path:
        base = self.project_root / "artifacts" / "datasets" / "snapshots"
        return base if self.collection is None else base / self.collection

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
            self.downloads,
            self.generated,
            self.sources,
            self.snapshots,
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

    def require_usable(self, identifier: str, *, local_research_only: bool) -> LicenseRule:
        if not identifier.strip():
            raise ValueError("missing license or terms label")
        rule = self.rule_for(identifier)
        if rule is None:
            raise ValueError(f"unknown license or terms label: {identifier}")
        if rule.status == "denied":
            raise ValueError(f"license or terms label is denied: {identifier}")
        if rule.status == "review_required" and not local_research_only:
            raise ValueError(f"license is not release-cleared: {identifier}")
        return rule


@dataclass(frozen=True, slots=True)
class PilotBudget:
    maximum_download_bytes: int
    maximum_raw_documents: int
    maximum_clean_documents: int
    maximum_estimated_tokens: int
    maximum_documents_per_source: int
    maximum_tokens_per_source: int
    maximum_archive_files: int
    maximum_decompressed_bytes: int
    request_timeout_seconds: int
    maximum_retries: int
    maximum_tokens_per_category: dict[str, int]
    minimum_tokens_per_category: dict[str, int]
    category_targets: dict[str, float]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PilotBudget:
        budget = cls(
            maximum_download_bytes=int(value["maximum_download_bytes"]),
            maximum_raw_documents=int(value["maximum_raw_documents"]),
            maximum_clean_documents=int(value["maximum_clean_documents"]),
            maximum_estimated_tokens=int(value["maximum_estimated_tokens"]),
            maximum_documents_per_source=int(value["maximum_documents_per_source"]),
            maximum_tokens_per_source=int(value["maximum_tokens_per_source"]),
            maximum_archive_files=int(value.get("maximum_archive_files", 10000)),
            maximum_decompressed_bytes=int(value.get("maximum_decompressed_bytes", 2_000_000_000)),
            request_timeout_seconds=int(value.get("request_timeout_seconds", 30)),
            maximum_retries=int(value.get("maximum_retries", 3)),
            maximum_tokens_per_category={name: int(limit) for name, limit in value["maximum_tokens_per_category"].items()},
            minimum_tokens_per_category={name: int(limit) for name, limit in value["minimum_tokens_per_category"].items()},
            category_targets={name: float(target) for name, target in value["category_targets"].items()},
        )
        budget.validate()
        return budget

    def validate(self) -> None:
        from cyber_agent.data_pipeline.schemas import CATEGORIES

        scalar_limits = (
            self.maximum_download_bytes,
            self.maximum_raw_documents,
            self.maximum_clean_documents,
            self.maximum_estimated_tokens,
            self.maximum_documents_per_source,
            self.maximum_tokens_per_source,
            self.maximum_archive_files,
            self.maximum_decompressed_bytes,
            self.request_timeout_seconds,
        )
        if any(limit < 1 for limit in scalar_limits):
            raise ValueError("pilot budget limits must be positive")
        if self.maximum_retries < 0:
            raise ValueError("maximum_retries must not be negative")
        if set(self.maximum_tokens_per_category) != CATEGORIES:
            raise ValueError("maximum_tokens_per_category must cover every category")
        if set(self.minimum_tokens_per_category) != CATEGORIES:
            raise ValueError("minimum_tokens_per_category must cover every category")
        if set(self.category_targets) != CATEGORIES:
            raise ValueError("category_targets must cover every category")
        if any(value < 0 for value in (*self.minimum_tokens_per_category.values(), *self.category_targets.values())):
            raise ValueError("pilot category minimums and targets must not be negative")
        if any(value < 1 for value in self.maximum_tokens_per_category.values()):
            raise ValueError("pilot category maximums must be positive")
        for category in CATEGORIES:
            if self.minimum_tokens_per_category[category] > self.maximum_tokens_per_category[category]:
                raise ValueError(f"pilot category minimum exceeds maximum: {category}")
        if abs(sum(self.category_targets.values()) - 1.0) > 1e-9:
            raise ValueError("pilot category targets must sum to 1.0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_download_bytes": self.maximum_download_bytes,
            "maximum_raw_documents": self.maximum_raw_documents,
            "maximum_clean_documents": self.maximum_clean_documents,
            "maximum_estimated_tokens": self.maximum_estimated_tokens,
            "maximum_documents_per_source": self.maximum_documents_per_source,
            "maximum_tokens_per_source": self.maximum_tokens_per_source,
            "maximum_archive_files": self.maximum_archive_files,
            "maximum_decompressed_bytes": self.maximum_decompressed_bytes,
            "request_timeout_seconds": self.request_timeout_seconds,
            "maximum_retries": self.maximum_retries,
            "maximum_tokens_per_category": dict(sorted(self.maximum_tokens_per_category.items())),
            "minimum_tokens_per_category": dict(sorted(self.minimum_tokens_per_category.items())),
            "category_targets": dict(sorted(self.category_targets.items())),
        }


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
    pilot_budget: PilotBudget
    research_budget: PilotBudget
    dataset_mode: DatasetMode

    @classmethod
    def load(cls, project_root: Path | None = None) -> PipelineConfig:
        paths = PipelinePaths((project_root or default_project_root()).resolve())
        settings = _load_json(paths.configuration / "data_pipeline.json")
        raw_policy = _load_json(paths.configuration / "license_policy.json")
        raw_budget = _load_json(paths.configuration / "pilot_budget.json")
        raw_research_budget = _load_json(paths.configuration / "research_budget.json")
        raw_mode = _load_json(paths.configuration / "dataset_mode.json")
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
            pilot_budget=PilotBudget.from_dict(raw_budget),
            research_budget=PilotBudget.from_dict(raw_research_budget),
            dataset_mode=DatasetMode.from_dict(raw_mode),
        )
        config.paths.ensure_directories()
        return config

    def for_collection(self, collection: str | None) -> PipelineConfig:
        """Return an isolated collection configuration without mutating pilot outputs.

        Named collections deliberately use the separately reviewed research budget.
        The default (``None``) retains the small fixture/pilot budget so existing
        snapshots and training manifests cannot be changed accidentally.
        """
        if collection is None:
            return self
        scoped_paths = PipelinePaths(self.paths.project_root, collection)
        scoped_paths.ensure_directories()
        return replace(self, paths=scoped_paths, pilot_budget=self.research_budget)

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

    def pilot_fingerprint_payload(self) -> dict[str, Any]:
        return {
            **self.fingerprint_payload(),
            "pilot_budget": self.pilot_budget.to_dict(),
            "collection": self.paths.collection,
            "dataset_mode": self.dataset_mode.to_dict(),
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a JSON object: {path}")
    return value
