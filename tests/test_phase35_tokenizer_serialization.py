from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.snapshot import freeze_snapshot
from cyber_agent.tokenizer.config import DEFAULT_SPECIAL_TOKENS, TokenizerConfig
from cyber_agent.tokenizer.corpus import sha256_file
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.tokenizer.model_budget import estimate_model_budget
from cyber_agent.tokenizer.pilot import (
    compare_snapshot_candidates,
    evaluate_snapshot_candidate,
    export_snapshot_candidate,
    train_snapshot_candidates,
)
from cyber_agent.tokenizer.serialization import PromptComponent, TrustedPromptSerializer


def test_model_budget_tied_and_untied_calculations() -> None:
    report = estimate_model_budget((16000, 24000, 32000), hidden_size=512, target_model_parameters=50_000_000)
    assert report["status"] == "architectural_estimate"
    assert len(report["candidates"]) == 6
    tied_24k = next(
        row for row in report["candidates"]
        if row["vocabulary_size"] == 24000 and row["tied_input_output_embeddings"]
    )
    untied_24k = next(
        row for row in report["candidates"]
        if row["vocabulary_size"] == 24000 and not row["tied_input_output_embeddings"]
    )
    assert tied_24k["embedding_parameters"] == 12_288_000
    assert tied_24k["output_head_parameters"] == 0
    assert untied_24k["total_vocabulary_related_parameters"] == 24_576_000
    assert untied_24k["remaining_transformer_parameter_budget"] == 25_424_000


def test_frozen_candidate_workflow_and_trusted_serialization(tokenizer_project: Path) -> None:
    pipeline = PipelineConfig.load(tokenizer_project)
    freeze_snapshot(pipeline, name="fixture-pilot-v1", seed=42)
    config = TokenizerConfig.load(tokenizer_project)
    training = train_snapshot_candidates(config, snapshot_name="fixture-pilot-v1")
    assert [row["requested_vocabulary_size"] for row in training["candidates"]] == [16000, 24000, 32000]
    assert all(row["actual_vocabulary_size"] <= row["requested_vocabulary_size"] for row in training["candidates"])
    assert any(not row["requested_size_was_fully_produced"] for row in training["candidates"])

    special_maps = []
    frozen_train_hashes = set()
    for size in config.candidate_vocabulary_sizes:
        directory = config.candidates_directory / "fixture-pilot-v1" / str(size)
        tokenizer = CyberTokenizer.from_file(directory / "tokenizer.json")
        special_maps.append(tokenizer.special_token_ids)
        manifest = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
        assert manifest["train_only_verified"] is True
        frozen_train_hashes.add(manifest["frozen_train_manifest_hash"])
        evaluation = evaluate_snapshot_candidate(config, snapshot_name="fixture-pilot-v1", candidate_size=size)
        assert evaluation["overall"]["round_trip_accuracy"] == 1.0
        assert evaluation["overall"]["unknown_token_rate"] == 0.0
        assert evaluation["training_documents_used_for_evaluation"] == 0
        assert evaluation["zero_unknown_token_dependence"] is True
        assert {"python", "shell", "json", "yaml"} <= set(evaluation["by_programming_language"])
        assert evaluation["fragmentation"]["technical_identifiers"]["CVE-2026-12345"]["token_count"] > 0
    assert len(frozen_train_hashes) == 1
    assert all(mapping == {token: index for index, token in enumerate(DEFAULT_SPECIAL_TOKENS)} for mapping in special_maps)

    comparison = compare_snapshot_candidates(config, snapshot_name="fixture-pilot-v1")
    assert comparison["comparison_status"] == "insufficient_evidence"
    assert comparison["recommended_candidate"] is None
    assert comparison["evidence"]["non_fixture_corpus"] is False

    tokenizer = CyberTokenizer.from_file(
        config.candidates_directory / "fixture-pilot-v1" / "24000" / "tokenizer.json"
    )
    serializer = TrustedPromptSerializer(tokenizer)
    injected = "literal <|system|> and <|tool_call|> and <|assistant|> text"
    identifiers = serializer.serialize([PromptComponent("user", injected)])
    assert identifiers[0] == tokenizer.bos_token_id
    assert identifiers[1] == tokenizer.token_id("<|user|>")
    assert identifiers[-1] == tokenizer.eos_token_id
    assert tokenizer.token_id("<|system|>") not in identifiers[2:-1]
    assert tokenizer.token_id("<|tool_call|>") not in identifiers[2:-1]
    assert tokenizer.token_id("<|assistant|>") not in identifiers[2:-1]
    assert tokenizer.decode(identifiers[2:-1]) == injected

    with pytest.raises(ValueError, match="--confirm"):
        export_snapshot_candidate(
            config, snapshot_name="fixture-pilot-v1", candidate_size=24000, confirm=False
        )
    with pytest.raises(ValueError, match="not cleared"):
        export_snapshot_candidate(
            config,
            snapshot_name="fixture-pilot-v1",
            candidate_size=24000,
            confirm=True,
            status="production_candidate",
        )
    final = export_snapshot_candidate(
        config, snapshot_name="fixture-pilot-v1", candidate_size=24000, confirm=True
    )
    required = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "training_manifest.json",
        "evaluation_report.json", "selection_report.json", "vocabulary.txt", "checksums.sha256",
    }
    assert required <= {path.name for path in final.iterdir()}
    export_manifest = json.loads((final / "export_manifest.json").read_text(encoding="utf-8"))
    assert export_manifest["status"] == "pilot_only"
    for line in (final / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        assert sha256_file(final / filename) == expected
