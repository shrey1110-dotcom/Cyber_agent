from __future__ import annotations

import json
from pathlib import Path

from cyber_agent.data_pipeline.cli import main
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.deduplicate import run_deduplicate
from cyber_agent.data_pipeline.export import read_jsonl, run_export
from cyber_agent.data_pipeline.ingest import run_ingest
from cyber_agent.data_pipeline.normalize import run_clean
from cyber_agent.data_pipeline.reports import run_report
from cyber_agent.data_pipeline.split import run_split


def test_end_to_end_fixture_processing_and_reports(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    ingest = run_ingest(config, ["sample"], force=True)
    clean = run_clean(config, force=True)
    deduplicate = run_deduplicate(config, force=True)
    split = run_split(config, seed=42, force=True)
    export = run_export(config, force=True)
    report = run_report(config)

    assert (ingest["accepted"], ingest["rejected"]) == (8, 2)
    assert (clean["accepted"], clean["rejected"]) == (6, 2)
    assert (deduplicate["retained"], deduplicate["exact"], deduplicate["near"]) == (4, 1, 1)
    assert split["train"] + split["validation"] + split["test"] == 4
    assert export["documents"] == 4
    assert report["accepted_documents"] == 4
    assert report["by_license"] == {"CC0-1.0": 4}
    assert report["rejections"]["total"] == 4
    assert report["duplicates"] == {"removed": 2, "exact": 1, "near": 1, "groups": 1}

    rejections = read_jsonl(config.paths.manifests / "rejection_manifest.jsonl")
    reason_codes = {code for record in rejections for code in record["reason_codes"]}
    assert {"license_rejected", "malformed_manifest", "aws_access_key", "pathological_repetition"} <= reason_codes
    serialized_rejections = json.dumps(rejections)
    assert "AKIAIOSFODNN7EXAMPLE" not in serialized_rejections

    code_document = next(
        record for record in read_jsonl(config.paths.cleaned / "dataset.jsonl") if record["category"] == "code"
    )
    assert '    completed = subprocess.run(' in code_document["text"]
    assert (config.paths.reports / "dataset_summary.json").exists()
    assert (config.paths.reports / "license_counts.json").exists()


def test_cli_returns_nonzero_for_unallowlisted_source(pipeline_project: Path, capsys) -> None:
    exit_code = main(
        [
            "--project-root",
            str(pipeline_project),
            "ingest",
            "--source",
            "not-approved",
        ]
    )
    assert exit_code == 1
    assert "not present in the allowlist" in capsys.readouterr().err

