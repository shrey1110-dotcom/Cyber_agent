from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import atomic_write_text
from cyber_agent.data_pipeline.ingest import run_ingest
from cyber_agent.data_pipeline.normalize import run_clean


def test_atomic_write_preserves_previous_output_on_interruption(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("previous-good-output", encoding="utf-8")

    def interrupt(temporary_path: Path) -> None:
        assert temporary_path.exists()
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        atomic_write_text(target, "incomplete-new-output", before_replace=interrupt)
    assert target.read_text(encoding="utf-8") == "previous-good-output"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corrupt_stage_marker_is_safely_recovered(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    first = run_ingest(config, ["sample"], force=True)
    assert first["status"] == "complete"
    marker = config.paths.manifests / "stages" / "ingest.json"
    marker.write_text("{interrupted", encoding="utf-8")
    second = run_ingest(config, ["sample"])
    assert second["status"] == "complete"
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "complete"


def test_completed_clean_stage_is_resumable(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    run_ingest(config, ["sample"], force=True)
    assert run_clean(config, force=True)["status"] == "complete"
    assert run_clean(config)["status"] == "skipped"

