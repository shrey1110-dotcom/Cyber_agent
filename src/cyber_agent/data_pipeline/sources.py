"""Approved-source registry and deliberately local-only source adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import LicensePolicy, PipelinePaths
from cyber_agent.data_pipeline.schemas import CATEGORIES, Category

REVIEW_STATUSES = frozenset({"pending", "approved_for_pilot", "approved_for_production", "rejected", "disabled"})
APPROVED_REVIEW_STATUSES = frozenset({"approved_for_pilot", "approved_for_production"})


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    source_name: str
    exact_release_or_version: str
    homepage: str
    download_location: str
    publisher: str
    data_location: str
    license: str
    license_evidence_url: str
    allowed_use: str
    redistribution_status: str
    attribution_requirements: str
    per_record_license_field: str
    review_status: str
    reviewed_by: str
    reviewed_at: str
    content_categories: tuple[Category, ...]
    known_risks: tuple[str, ...]
    category: Category
    adapter: str
    enabled: bool
    notes: str = ""
    published_sha256: str = ""
    approved_domains: tuple[str, ...] = ()
    retrieved_at: str = ""
    local_research_source: bool = False
    acquisition_enabled: bool = False
    maximum_download_bytes: int | None = None
    adapter_options: dict[str, Any] | None = None
    collection: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceDefinition:
        local_research_source = value.get("local_research_source", False)
        if not isinstance(local_research_source, bool):
            raise ValueError("local_research_source must be a boolean")
        base_required = {
            "source_name", "exact_release_or_version", "download_location",
            "publisher", "license", "category", "retrieved_at",
        }
        detailed_required = {
            "source_name",
            "exact_release_or_version",
            "homepage",
            "download_location",
            "publisher",
            "data_location",
            "license",
            "license_evidence_url",
            "allowed_use",
            "redistribution_status",
            "attribution_requirements",
            "per_record_license_field",
            "review_status",
            "reviewed_by",
            "reviewed_at",
            "content_categories",
            "known_risks",
            "category",
            "adapter",
            "enabled",
        }
        required = base_required if local_research_source else detailed_required
        missing = required - set(value)
        if missing:
            raise ValueError(f"source is missing fields: {', '.join(sorted(missing))}")
        converted = {
            "source_name": value["source_name"],
            "exact_release_or_version": value["exact_release_or_version"],
            "homepage": value.get("homepage", value["download_location"]),
            "download_location": value["download_location"],
            "publisher": value["publisher"],
            "data_location": value.get("data_location", f"data/sources/{value['source_name']}/manifest.jsonl"),
            "license": value["license"],
            "license_evidence_url": value.get("license_evidence_url", "local-research://stated-terms-not-verified"),
            "allowed_use": value.get("allowed_use", "Private local research only; not cleared for release."),
            "redistribution_status": value.get("redistribution_status", "Not cleared for redistribution."),
            "attribution_requirements": value.get("attribution_requirements", "Preserve source, publisher, URL, retrieval date, and stated terms."),
            "per_record_license_field": value.get("per_record_license_field", "license"),
            "review_status": value.get("review_status", "pending"),
            "reviewed_by": value.get("reviewed_by", ""),
            "reviewed_at": value.get("reviewed_at", ""),
            "content_categories": value.get("content_categories", [value["category"]]),
            "known_risks": value.get("known_risks", ["Not reviewed for public dataset or model-weight release."]),
            "category": value["category"],
            "adapter": value.get("adapter", "http_archive_text"),
            "enabled": value.get("enabled", False),
        }
        if not isinstance(converted["content_categories"], list) or not isinstance(converted["known_risks"], list):
            raise ValueError("content_categories and known_risks must be arrays")
        converted["content_categories"] = tuple(converted["content_categories"])
        converted["known_risks"] = tuple(converted["known_risks"])
        approved_domains = value.get("approved_domains", [])
        if not isinstance(approved_domains, list) or any(not isinstance(domain, str) for domain in approved_domains):
            raise ValueError("approved_domains must be an array of domain names")
        return cls(
            **converted,
            notes=value.get("notes", ""),
            published_sha256=value.get("published_sha256", ""),
            approved_domains=tuple(approved_domains),
            retrieved_at=value.get("retrieved_at", value.get("reviewed_at", "")),
            local_research_source=local_research_source,
            acquisition_enabled=value.get("acquisition_enabled", False),
            maximum_download_bytes=(
                int(value["maximum_download_bytes"])
                if value.get("maximum_download_bytes") is not None else None
            ),
            adapter_options=value.get("adapter_options", {}),
            collection=value.get("collection", ""),
        )

    def validate(self, license_policy: LicensePolicy) -> list[str]:
        errors: list[str] = []
        for field_name in (
            "source_name",
            "exact_release_or_version",
            "homepage",
            "download_location",
            "publisher",
            "data_location",
            "license",
            "license_evidence_url",
            "allowed_use",
            "redistribution_status",
            "attribution_requirements",
            "per_record_license_field",
            "adapter",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name).strip():
                errors.append(f"{self.source_name or '<unnamed>'}: {field_name} is required")
        if self.category not in CATEGORIES:
            errors.append(f"{self.source_name}: unsupported category {self.category}")
        if self.collection and any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in self.collection
        ):
            errors.append(f"{self.source_name}: collection name contains unsupported characters")
        if not self.content_categories or any(category not in CATEGORIES for category in self.content_categories):
            errors.append(f"{self.source_name}: content_categories must contain supported categories")
        if self.category not in self.content_categories:
            errors.append(f"{self.source_name}: primary category must appear in content_categories")
        if self.review_status not in REVIEW_STATUSES:
            errors.append(f"{self.source_name}: unsupported review status {self.review_status}")
        if self.published_sha256:
            digest = self.published_sha256.casefold()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                errors.append(f"{self.source_name}: published_sha256 is not a full SHA-256 digest")
        rule = license_policy.rule_for(self.license)
        if rule is None:
            errors.append(f"{self.source_name}: license is absent from policy: {self.license}")
        approved = self.review_status in APPROVED_REVIEW_STATUSES
        if self.local_research_source:
            if not self.retrieved_at.strip():
                errors.append(f"{self.source_name}: retrieval date is required")
            else:
                try:
                    retrieved = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
                    if retrieved.tzinfo is None:
                        raise ValueError
                except ValueError:
                    errors.append(f"{self.source_name}: retrieved_at must be timezone-aware ISO-8601")
            if rule is not None and rule.status == "denied":
                errors.append(f"{self.source_name}: denied terms cannot be used in local research mode")
            if self.acquisition_enabled and self.adapter.startswith("http_") and not self.approved_domains:
                errors.append(f"{self.source_name}: remote source requires approved_domains")
            if self.maximum_download_bytes is not None and self.maximum_download_bytes < 1:
                errors.append(f"{self.source_name}: maximum_download_bytes must be positive when configured")
        if self.enabled and not approved and not self.local_research_source:
            errors.append(f"{self.source_name}: enabled source is not approved for pilot or production")
        if approved and (not self.reviewed_by.strip() or not self.reviewed_at.strip()):
            errors.append(f"{self.source_name}: approved source requires reviewer and review timestamp")
        if approved and self.exact_release_or_version == "UNPINNED":
            errors.append(f"{self.source_name}: approved source requires an exact pinned release")
        if approved and self.reviewed_at.strip():
            try:
                reviewed = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
                if reviewed.tzinfo is None:
                    raise ValueError
            except ValueError:
                errors.append(f"{self.source_name}: reviewed_at must be timezone-aware ISO-8601")
        # A local-research source may be explicitly reviewed for bounded private
        # experimentation even when its terms still need release review (for
        # example CC-BY-SA attribution/share-alike obligations).  It remains
        # blocked from audited-release mode by DatasetMode and snapshot notices.
        approval_usable = rule is not None and (
            rule.status == "allowed"
            or (self.local_research_source and rule.status == "review_required")
        )
        if approved and not approval_usable:
            errors.append(
                f"{self.source_name}: approved source must have allowed terms "
                "or explicitly local-research-usable review-required terms"
            )
        if self.enabled and not self.local_research_source and (rule is None or rule.status != "allowed"):
            errors.append(f"{self.source_name}: enabled source must have an allowed license")
        return errors

    @property
    def is_approved(self) -> bool:
        return self.review_status in APPROVED_REVIEW_STATUSES


class SourceRegistry:
    def __init__(self, definitions: list[SourceDefinition], paths: PipelinePaths) -> None:
        self.paths = paths
        self._sources = {source.source_name: source for source in definitions}
        if len(self._sources) != len(definitions):
            raise ValueError("source names must be unique")

    @classmethod
    def load(cls, paths: PipelinePaths) -> SourceRegistry:
        config_path = paths.configuration / "approved_sources.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load source allowlist {config_path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
            raise ValueError("approved_sources.json must contain a sources array")
        definitions = [SourceDefinition.from_dict(item) for item in payload["sources"]]
        mode_path = paths.configuration / "dataset_mode.json"
        local_path = paths.configuration / "local_research_sources.json"
        if mode_path.exists() and local_path.exists():
            try:
                mode = json.loads(mode_path.read_text(encoding="utf-8"))
                local_payload = json.loads(local_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot load local research source configuration: {exc}") from exc
            if mode.get("dataset_mode") == "local_research_only":
                if not isinstance(local_payload, dict) or not isinstance(local_payload.get("sources"), list):
                    raise ValueError("local_research_sources.json must contain a sources array")
                definitions.extend(SourceDefinition.from_dict(item) for item in local_payload["sources"])
        return cls(definitions, paths)

    def validate(self, license_policy: LicensePolicy) -> list[str]:
        return [error for source in self._sources.values() for error in source.validate(license_policy)]

    def require_ingestible(self, source_name: str, license_policy: LicensePolicy) -> SourceDefinition:
        source = self._sources.get(source_name)
        if source is None:
            raise ValueError(f"source is not present in the allowlist: {source_name}")
        errors = source.validate(license_policy)
        if errors:
            raise ValueError("; ".join(errors))
        if not source.enabled:
            raise ValueError(f"source is configured as a disabled placeholder: {source_name}")
        self._require_collection(source)
        if not source.is_approved:
            raise ValueError(f"source review status does not permit ingestion: {source_name} ({source.review_status})")
        if source.adapter not in {
            "local_manifest", "synthetic_tool_examples", "http_archive_text",
            "http_stix_json", "http_cwe_xml", "http_wikimedia_xml_bz2",
        }:
            raise ValueError(f"source adapter is not an ingestible local manifest: {source.adapter}")
        return source

    def require_downloadable(self, source_name: str, license_policy: LicensePolicy) -> SourceDefinition:
        source = self._sources.get(source_name)
        if source is None:
            raise ValueError(f"source is not present in the allowlist: {source_name}")
        errors = source.validate(license_policy)
        if errors:
            raise ValueError("; ".join(errors))
        if not source.enabled or not source.is_approved:
            raise ValueError(f"source is not approved for download: {source_name} ({source.review_status})")
        self._require_collection(source)
        if source.adapter not in {
            "http_file", "http_archive", "http_archive_text", "http_stix_json",
            "http_cwe_xml", "http_wikimedia_xml_bz2",
        }:
            raise ValueError(f"source has no remote download adapter: {source_name}")
        return source

    def enabled_sources(self, license_policy: LicensePolicy) -> list[SourceDefinition]:
        return [
            self.require_ingestible(source.source_name, license_policy)
            for source in self._sources.values()
            if source.enabled and self._matches_collection(source)
        ]

    def acquisition_sources(self, license_policy: LicensePolicy) -> list[SourceDefinition]:
        return [
            self.require_downloadable(source.source_name, license_policy)
            for source in self._sources.values()
            if source.enabled
            and source.is_approved
            and source.local_research_source
            and source.acquisition_enabled
            and source.adapter.startswith("http_")
            and self._matches_collection(source)
        ]

    def synthetic_sources(self) -> list[SourceDefinition]:
        return [
            source for source in self._sources.values()
            if source.enabled
            and source.is_approved
            and source.local_research_source
            and source.acquisition_enabled
            and source.adapter == "synthetic_tool_examples"
            and self._matches_collection(source)
        ]

    def all_sources(self) -> list[SourceDefinition]:
        return list(self._sources.values())

    def source_by_name(self, source_name: str) -> SourceDefinition:
        """Return a configured source for safe same-release archive reuse checks."""
        try:
            return self._sources[source_name]
        except KeyError as exc:
            raise ValueError(f"source is not present in the allowlist: {source_name}") from exc

    def manifest_path(self, source: SourceDefinition) -> Path:
        path = (self.paths.project_root / source.data_location).resolve()
        try:
            path.relative_to(self.paths.project_root)
        except ValueError as exc:
            raise ValueError(f"source manifest resolves outside project: {source.source_name}") from exc
        return path

    def _matches_collection(self, source: SourceDefinition) -> bool:
        return source.collection == (self.paths.collection or "")

    def _require_collection(self, source: SourceDefinition) -> None:
        if not self._matches_collection(source):
            configured = source.collection or "default"
            active = self.paths.collection or "default"
            raise ValueError(f"source is configured for collection {configured}, not active collection {active}")
