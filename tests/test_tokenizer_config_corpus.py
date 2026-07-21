from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.schemas import sha256_text
from cyber_agent.tokenizer.config import DEFAULT_SPECIAL_TOKENS, TokenizerConfig
from cyber_agent.tokenizer.corpus import validate_training_corpus


def test_tokenizer_configuration_and_corpus_manifest(tokenizer_project: Path) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=512)
    corpus = validate_training_corpus(config)
    assert config.algorithm == "byte_level_bpe"
    assert config.special_tokens == DEFAULT_SPECIAL_TOKENS
    assert corpus.document_count == 4
    assert corpus.source_counts == {"sample": 4}
    assert corpus.license_counts == {"CC0-1.0": 4}
    assert corpus.excluded_split_counts == {"validation": 0, "test": 0}
    assert len(corpus.manifest_documents()) == 4


@pytest.mark.parametrize("split_name", ["validation", "test"])
def test_training_split_only_enforcement(tokenizer_project: Path, split_name: str) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=512)
    forbidden = replace(config, training_split_path=tokenizer_project / "data" / "splits" / f"{split_name}.jsonl")
    with pytest.raises(ValueError, match="restricted to the Phase 2 training split"):
        validate_training_corpus(forbidden)


def test_missing_required_manifest_fails_closed(tokenizer_project: Path) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=512)
    (tokenizer_project / "data" / "manifests" / "train_manifest.jsonl").unlink()
    with pytest.raises(ValueError, match="required Phase 2 tokenizer inputs are missing"):
        validate_training_corpus(config)


def test_invalid_license_metadata_is_rejected(tokenizer_project: Path) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=512)
    source_manifest = tokenizer_project / "data" / "manifests" / "source_manifest.jsonl"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source["license"] = "LicenseRef-Unknown"
    source_manifest.write_text(json.dumps(source) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown license"):
        validate_training_corpus(config)


def test_empty_training_record_is_rejected(tokenizer_project: Path) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=512)
    split_path = tokenizer_project / "data" / "splits" / "train.jsonl"
    rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = ""
    rows[0]["content_hash"] = sha256_text("")
    split_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = tokenizer_project / "data" / "manifests" / "train_manifest.jsonl"
    assignments = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assignments[0]["content_hash"] = rows[0]["content_hash"]
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in assignments), encoding="utf-8")
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        validate_training_corpus(config)


def test_very_long_training_record_is_rejected(tokenizer_project: Path) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=512)
    strict = replace(config, maximum_document_characters=100)
    with pytest.raises(ValueError, match="exceeds configured length"):
        validate_training_corpus(strict)

