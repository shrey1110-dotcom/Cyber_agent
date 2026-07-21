from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_agent.tokenizer.artifacts import CORE_ARTIFACT_NAMES, deterministic_artifact_hashes
from cyber_agent.tokenizer.config import DEFAULT_SPECIAL_TOKENS, TokenizerConfig
from cyber_agent.tokenizer.corpus import sha256_file
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.tokenizer.trainer import train_candidate


def train_fixture(project: Path, *, seed: int = 42) -> tuple[TokenizerConfig, CyberTokenizer]:
    config = TokenizerConfig.load(project).with_overrides(vocabulary_size=512, seed=seed)
    train_candidate(config, fixture_artifact=True)
    return config, CyberTokenizer.from_file(config.candidate_directory / "tokenizer.json")


def test_deterministic_training_and_complete_artifacts(tokenizer_project: Path) -> None:
    config, _ = train_fixture(tokenizer_project)
    first_hashes = deterministic_artifact_hashes(config.candidate_directory)
    train_candidate(config, fixture_artifact=True)
    second_hashes = deterministic_artifact_hashes(config.candidate_directory)
    assert first_hashes == second_hashes
    assert set(first_hashes) == set(CORE_ARTIFACT_NAMES)
    for required in (*CORE_ARTIFACT_NAMES, "training_manifest.json", "FIXTURE_ONLY.txt"):
        assert (config.candidate_directory / required).exists()
    manifest = json.loads((config.candidate_directory / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fixture_artifact"] is True
    assert manifest["production_ready"] is False
    assert manifest["input_document_count"] == 4
    assert manifest["excluded_split_document_counts"] == {"validation": 0, "test": 0}
    assert len(manifest["training_documents"]) == 4


def test_different_seed_is_recorded(tokenizer_project: Path) -> None:
    config, _ = train_fixture(tokenizer_project, seed=99)
    manifest = json.loads((config.candidate_directory / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 99
    assert manifest["configuration_hash"] == config.configuration_hash()


def test_training_does_not_modify_phase2_inputs(tokenizer_project: Path) -> None:
    paths = [
        tokenizer_project / "data" / "splits" / f"{name}.jsonl"
        for name in ("train", "validation", "test")
    ] + [
        tokenizer_project / "data" / "manifests" / f"{name}_manifest.jsonl"
        for name in ("train", "validation", "test")
    ]
    before = {path: sha256_file(path) for path in paths}
    train_fixture(tokenizer_project)
    assert {path: sha256_file(path) for path in paths} == before


def test_stable_special_ids_and_literal_special_token_safety(tokenizer_project: Path) -> None:
    _, tokenizer = train_fixture(tokenizer_project)
    assert tokenizer.special_token_ids == {token: index for index, token in enumerate(DEFAULT_SPECIAL_TOKENS)}
    for token, identifier in tokenizer.special_token_ids.items():
        literal_ids = tokenizer.encode(token)
        assert literal_ids != [identifier]
        assert tokenizer.decode(literal_ids) == token
        assert tokenizer.encode(token, parse_special_tokens=True) == [identifier]


@pytest.mark.parametrize(
    "text",
    [
        "Résumé integrity — 東京 — 🔐",
        "journalctl -u ssh --since \"2 hours ago\"",
        "grep -R \"failed password\" /var/log && echo done | sort >> audit.log || exit 1",
        "/var/log/auth.log ../../etc/passwd ~/.ssh/id_ed25519",
        "192.168.1.10:443 2001:db8::1",
        "CVE-2026-12345 CWE-79 T1059.004",
        '{"type":"tool_call","tool":"check_ports","arguments":{}}',
        "service:\n  ports:\n    - 22\n  enabled: true",
        "def validate(path):\n    if path:\n        return path.resolve()",
    ],
)
def test_byte_level_utf8_and_structured_text_round_trip(tokenizer_project: Path, text: str) -> None:
    _, tokenizer = train_fixture(tokenizer_project)
    token_ids = tokenizer.encode(text)
    assert tokenizer.unk_token_id not in token_ids
    assert tokenizer.decode(token_ids) == text


def test_batch_bos_eos_padding_attention_and_truncation(tokenizer_project: Path) -> None:
    _, tokenizer = train_fixture(tokenizer_project)
    encoded = tokenizer.encode("check port 8080", add_bos=True, add_eos=True, maximum_length=8)
    assert len(encoded) == 8
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id
    batch = tokenizer.batch_encode(["short", "a longer value"], add_bos=True, add_eos=True, padding=True)
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert batch["attention_mask"][0][-1] == 0
    assert batch["attention_mask"][1][-1] == 1


def test_production_sized_training_is_blocked_on_fixture(tokenizer_project: Path) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=24000)
    with pytest.raises(ValueError, match="too small for this vocabulary"):
        train_candidate(config, fixture_artifact=True)

