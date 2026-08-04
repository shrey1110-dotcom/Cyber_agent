from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.sources import SourceRegistry


def _write_selection(project: Path, entries: list[dict[str, str]]) -> Path:
    directory = project / "config" / "collections"
    directory.mkdir(exist_ok=True)
    path = directory / "research-v2.json"
    path.write_text(json.dumps({"schema_version": 1, "collection": "research-v2", "sources": entries}), encoding="utf-8")
    return path


def test_collection_selection_clones_only_exact_reviewed_sources(pipeline_project: Path) -> None:
    _write_selection(
        pipeline_project,
        [{
            "source_name": "sample",
            "exact_release_or_version": "fixture-pilot-v1",
            "download_location": "local://fixtures/sample_corpus/manifest.jsonl",
        }],
    )
    config = PipelineConfig.load(pipeline_project).for_collection("research-v2")
    registry = SourceRegistry.load(config.paths)

    source = registry.require_ingestible("sample", config.license_policy)
    assert [item.source_name for item in registry.all_sources()] == ["sample"]
    assert source.collection == "research-v2"
    assert registry.manifest_path(source) == config.paths.sources / "sample" / "manifest.jsonl"


def test_collection_selection_rejects_review_release_drift(pipeline_project: Path) -> None:
    _write_selection(
        pipeline_project,
        [{
            "source_name": "sample",
            "exact_release_or_version": "fixture-pilot-v2",
            "download_location": "local://fixtures/sample_corpus/manifest.jsonl",
        }],
    )
    config = PipelineConfig.load(pipeline_project).for_collection("research-v2")
    with pytest.raises(ValueError, match="does not match review record"):
        SourceRegistry.load(config.paths)


def test_collection_can_use_a_reviewed_budget_profile(pipeline_project: Path) -> None:
    _write_selection(
        pipeline_project,
        [{
            "source_name": "sample",
            "exact_release_or_version": "fixture-pilot-v1",
            "download_location": "local://fixtures/sample_corpus/manifest.jsonl",
        }],
    )
    (pipeline_project / "config" / "collections" / "research-v2.json").write_text(
        json.dumps({
            "schema_version": 1,
            "collection": "research-v2",
            "budget_profile": "large_budget.json",
            "sources": [{
                "source_name": "sample",
                "exact_release_or_version": "fixture-pilot-v1",
                "download_location": "local://fixtures/sample_corpus/manifest.jsonl",
            }],
        }),
        encoding="utf-8",
    )
    (pipeline_project / "config" / "large_budget.json").write_text(
        json.dumps({
            "maximum_download_bytes": 123456,
            "maximum_raw_documents": 200,
            "maximum_clean_documents": 150,
            "maximum_estimated_tokens": 10000,
            "maximum_documents_per_source": 100,
            "maximum_tokens_per_source": 9000,
            "maximum_tokens_per_category": {name: 10000 for name in ("general", "code", "linux", "networking", "cybersecurity", "terminal")},
            "minimum_tokens_per_category": {name: 0 for name in ("general", "code", "linux", "networking", "cybersecurity", "terminal")},
            "category_targets": {
                "general": 0.3, "code": 0.25, "linux": 0.15,
                "networking": 0.1, "cybersecurity": 0.15, "terminal": 0.05,
            },
        }),
        encoding="utf-8",
    )

    config = PipelineConfig.load(pipeline_project).for_collection("research-v2")

    assert config.pilot_budget.maximum_estimated_tokens == 10_000
    assert config.pilot_budget.maximum_download_bytes == 123_456


def test_collection_rejects_budget_profile_path_escape(pipeline_project: Path) -> None:
    _write_selection(
        pipeline_project,
        [{
            "source_name": "sample",
            "exact_release_or_version": "fixture-pilot-v1",
            "download_location": "local://fixtures/sample_corpus/manifest.jsonl",
        }],
    )
    selection = pipeline_project / "config" / "collections" / "research-v2.json"
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["budget_profile"] = "../research_budget.json"
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="simple JSON filename"):
        PipelineConfig.load(pipeline_project).for_collection("research-v2")
