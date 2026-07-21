from __future__ import annotations

import json
from pathlib import Path

from cyber_agent.tokenizer.artifacts import export_final_tokenizer
from cyber_agent.tokenizer.cli import main
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.evaluator import compare_candidates, evaluate_tokenizer
from cyber_agent.tokenizer.trainer import train_candidate


def prepared_candidate(project: Path) -> TokenizerConfig:
    config = TokenizerConfig.load(project).with_overrides(vocabulary_size=512)
    train_candidate(config, fixture_artifact=True)
    return config


def test_evaluation_report_and_no_training_evaluation_leakage(tokenizer_project: Path) -> None:
    config = prepared_candidate(tokenizer_project)
    report = evaluate_tokenizer(config, config.candidate_directory / "tokenizer.json")
    assert report["overall"]["unknown_token_rate"] == 0.0
    assert report["overall"]["round_trip_accuracy"] == 1.0
    assert report["phase2_training_documents_used_for_evaluation"] == 0
    assert report["special_token_behavior"]["all_literal_text_safe"] is True
    assert report["special_token_behavior"]["all_explicit_parsing_correct"] is True
    assert {
        "english_prose",
        "python",
        "bash",
        "json",
        "yaml",
        "logs",
        "cybersecurity_identifiers",
        "paths",
        "network_addresses",
    } <= set(report["by_evaluation_group"])
    assert (config.candidate_directory / "evaluation_report.json").exists()


def test_candidate_comparison_does_not_auto_select_fixture(tokenizer_project: Path) -> None:
    config = prepared_candidate(tokenizer_project)
    evaluate_tokenizer(config, config.candidate_directory / "tokenizer.json")
    report = compare_candidates(config)
    assert report["comparison_status"] == "insufficient_test_data"
    assert report["automatic_selection"] is None
    assert report["missing_standard_candidates"] == [16000, 24000, 32000]


def test_final_export_requires_explicit_evaluated_candidate(tokenizer_project: Path) -> None:
    config = prepared_candidate(tokenizer_project)
    assert not (config.final_directory / "tokenizer.json").exists()
    evaluate_tokenizer(config, config.candidate_directory / "tokenizer.json")
    final_directory = export_final_tokenizer(config, 512)
    assert (final_directory / "tokenizer.json").exists()
    selection = json.loads((final_directory / "selection_manifest.json").read_text(encoding="utf-8"))
    assert selection["selection_is_explicit"] is True


def test_cli_errors_and_inspection(tokenizer_project: Path, capsys) -> None:
    config = prepared_candidate(tokenizer_project)
    failure = main(
        [
            "--project-root",
            str(tokenizer_project),
            "train",
            "--vocab-size",
            "24000",
            "--fixture",
        ]
    )
    assert failure == 1
    assert "too small" in capsys.readouterr().err
    success = main(
        [
            "--project-root",
            str(tokenizer_project),
            "inspect",
            "CVE-2026-12345",
            "--tokenizer",
            str(config.candidate_directory / "tokenizer.json"),
        ]
    )
    assert success == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decoded"] == "CVE-2026-12345"

