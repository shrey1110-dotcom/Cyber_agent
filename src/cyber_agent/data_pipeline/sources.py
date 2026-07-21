"""Approved-source registry and deliberately local-only source adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import LicensePolicy, PipelinePaths
from cyber_agent.data_pipeline.schemas import CATEGORIES, Category


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    source_name: str
    homepage: str
    data_location: str
    license: str
    allowed_use: str
    attribution_requirements: str
    category: Category
    adapter: str
    enabled: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceDefinition:
        required = {
            "source_name",
            "homepage",
            "data_location",
            "license",
            "allowed_use",
            "attribution_requirements",
            "category",
            "adapter",
            "enabled",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"source is missing fields: {', '.join(sorted(missing))}")
        return cls(**{name: value[name] for name in required}, notes=value.get("notes", ""))

    def validate(self, license_policy: LicensePolicy) -> list[str]:
        errors: list[str] = []
        for field_name in (
            "source_name",
            "homepage",
            "data_location",
            "license",
            "allowed_use",
            "attribution_requirements",
            "adapter",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name).strip():
                errors.append(f"{self.source_name or '<unnamed>'}: {field_name} is required")
        if self.category not in CATEGORIES:
            errors.append(f"{self.source_name}: unsupported category {self.category}")
        rule = license_policy.rule_for(self.license)
        if rule is None:
            errors.append(f"{self.source_name}: license is absent from policy: {self.license}")
        if self.enabled and (rule is None or rule.status != "allowed"):
            errors.append(f"{self.source_name}: enabled source must have an allowed license")
        if self.enabled and self.adapter != "local_manifest":
            errors.append(f"{self.source_name}: Phase 2 enables only local_manifest adapters")
        return errors


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
        return cls([SourceDefinition.from_dict(item) for item in payload["sources"]], paths)

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
        return source

    def enabled_sources(self, license_policy: LicensePolicy) -> list[SourceDefinition]:
        return [
            self.require_ingestible(source.source_name, license_policy)
            for source in self._sources.values()
            if source.enabled
        ]

    def all_sources(self) -> list[SourceDefinition]:
        return list(self._sources.values())

    def manifest_path(self, source: SourceDefinition) -> Path:
        path = (self.paths.project_root / source.data_location).resolve()
        try:
            path.relative_to(self.paths.project_root)
        except ValueError as exc:
            raise ValueError(f"source manifest resolves outside project: {source.source_name}") from exc
        return path

