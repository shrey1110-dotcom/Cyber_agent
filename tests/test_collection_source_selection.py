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
