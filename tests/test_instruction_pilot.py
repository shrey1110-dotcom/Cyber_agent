from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_agent.training.instruction import InstructionRecord, InstructionTrainingArtifacts


def _base(tokenizer) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=tokenizer,
        provenance=lambda: {
            "snapshot_name": "fixture-pilot-v1",
            "tokenizer_path": "artifacts/tokenizer.json",
            "tokenizer_hash": "a" * 64,
            "train_only_verified": True,
            "local_research_only": True,
            "weight_publication_allowed": False,
        },
    )


def test_instruction_pilot_is_reviewed_split_isolated_and_plain_text(tokenizer_project: Path) -> None:
    from cyber_agent.tokenizer.config import TokenizerConfig
    from cyber_agent.tokenizer.loader import CyberTokenizer
    from cyber_agent.tokenizer.trainer import train_candidate

    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=300)
    train_candidate(config, fixture_artifact=True)
    tokenizer = CyberTokenizer.from_file(config.candidates_directory / "300" / "tokenizer.json")
    artifacts = InstructionTrainingArtifacts.load(
        project_root=tokenizer_project,
        base=_base(tokenizer),
        dataset_path=tokenizer_project / "fixtures" / "instruction_pilot_v0.jsonl",
    )

    assert artifacts.provenance()["data_mode"] == "local_instruction_adaptation"
    assert artifacts.provenance()["instruction_split_counts"] == {"train": 32, "validation": 8}
    record = next(iter(artifacts._records("train")))
    ids = tokenizer.encode(record.render(), add_bos=True, parse_special_tokens=False)
    assert tokenizer.token_id("<|assistant|>") not in ids
    inputs, targets, weights = artifacts._supervised_example(record, sequence_length=512)
    assert len(inputs) == len(targets) == len(weights) == 512
    assert weights[0] == 0.0
    assert weights.count(1.0) > 1
    assert weights[-1] == 0.0


def test_instruction_pilot_rejects_missing_license_and_unapproved_source() -> None:
    record = {
        "record_id": "bad",
        "split": "train",
        "category": "english",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "source_name": "unknown",
    }
    with pytest.raises(ValueError, match="missing fields: license"):
        InstructionRecord.from_dict(record, line_number=1)

    record["license"] = "MIT"
    with pytest.raises(ValueError, match="unapproved source_name"):
        InstructionRecord.from_dict(record, line_number=1)


def test_instruction_pilot_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "instructions.jsonl"
    row = {
        "record_id": "same",
        "split": "train",
        "category": "english",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "license": "MIT",
        "source_name": "cyber-agent-authored-instruction-pilot",
    }
    second = {**row, "split": "validation"}
    path.write_text("\n".join(json.dumps(value) for value in (row, second)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate record_id"):
        InstructionTrainingArtifacts.load(project_root=tmp_path, base=SimpleNamespace(), dataset_path=path)
