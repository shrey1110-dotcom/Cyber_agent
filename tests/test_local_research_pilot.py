from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.pilot_acquisition import acquire_pilot
from cyber_agent.data_pipeline.sources import SourceDefinition, SourceRegistry
from cyber_agent.data_pipeline.synthetic import EXAMPLE_KINDS, SAFE_TOOLS, generate_safe_tool_examples


def local_source(**overrides):
    value = {
        "source_name": "local-review-source",
        "exact_release_or_version": "dataset-v1",
        "download_location": "https://approved.example/dataset-v1.zip",
        "publisher": "Example publisher",
        "license": "REVIEW_REQUIRED",
        "category": "cybersecurity",
        "retrieved_at": "2026-07-21T00:00:00-07:00",
        "approved_domains": ["approved.example"],
        "local_research_source": True,
        "acquisition_enabled": True,
    }
    value.update(overrides)
    return value


def test_local_research_mode_and_basic_source_metadata(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    assert config.dataset_mode.to_dict() == {
        "dataset_mode": "local_research_only",
        "release_cleared": False,
        "weight_publication_allowed": False,
        "dataset_redistribution_allowed": False,
    }
    source = SourceDefinition.from_dict(local_source())
    assert source.validate(config.license_policy) == []
    assert config.license_policy.require_usable("REVIEW_REQUIRED", local_research_only=True).status == "review_required"
    missing_license = local_source()
    missing_license.pop("license")
    with pytest.raises(ValueError, match="missing fields: license"):
        SourceDefinition.from_dict(missing_license)


def test_pilot_download_requires_one_explicit_confirmation(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    with pytest.raises(ValueError, match="--confirm-download"):
        acquire_pilot(
            config,
            mode="local_research_only",
            target_tokens=1_000_000,
            seed=42,
            confirm_download=False,
        )
    assert not (config.paths.manifests / "pilot_acquisition.json").exists()


def test_synthetic_tool_examples_are_deterministic_and_bounded(tmp_path: Path) -> None:
    source = SourceDefinition.from_dict(local_source(
        source_name="synthetic-test",
        exact_release_or_version="generator-v1",
        download_location="local://synthetic-test",
        publisher="Test generator",
        license="CC0-1.0",
        category="terminal",
        adapter="synthetic_tool_examples",
        approved_domains=[],
    ))
    count = len(SAFE_TOOLS) * len(EXAMPLE_KINDS)
    first = tmp_path / "first" / "manifest.jsonl"
    second = tmp_path / "second" / "manifest.jsonl"
    generate_safe_tool_examples(source, first, seed=42, document_count=count)
    generate_safe_tool_examples(source, second, seed=42, document_count=count)
    assert first.read_bytes() == second.read_bytes()
    first_documents = sorted((first.parent / "documents").glob("*.json"))
    second_documents = sorted((second.parent / "documents").glob("*.json"))
    assert [path.read_bytes() for path in first_documents] == [path.read_bytes() for path in second_documents]
    records = [json.loads(path.read_text()) for path in first_documents]
    assert all(record["provenance"]["synthetic"] is True for record in records)
    valid_calls = [record["input"] for record in records if record["provenance"]["example_kind"] == "valid_tool_call"]
    assert {call["tool"] for call in valid_calls} == set(SAFE_TOOLS)
    assert all(call["tool"] != "run_shell" for call in valid_calls)


def test_local_source_registry_visibly_reports_review_required(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    registry = SourceRegistry.load(config.paths)
    review_required = {
        source.source_name for source in registry.all_sources()
        if source.local_research_source and source.license == "REVIEW_REQUIRED"
    }
    assert {"linux-man-pages-6.15", "mitre-attack-stix-v17.1", "cwe-4.17"} <= review_required
