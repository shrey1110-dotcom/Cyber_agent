"""Transactional tokenizer artifact writing, hashing, and explicit final export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from cyber_agent.data_pipeline.export import atomic_write_json, atomic_write_text
from cyber_agent.data_pipeline.schemas import canonical_json, utc_now
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.corpus import CorpusPlan, sha256_file


SPECIAL_TOKEN_PURPOSES = {
    "<|pad|>": "Batch padding; ignored by attention masks and future loss masking.",
    "<|bos|>": "Beginning of a model sequence.",
    "<|eos|>": "End of a model sequence or completed answer.",
    "<|unk|>": "Unexpected fallback token; byte coverage should keep its normal-text rate at zero.",
    "<|system|>": "Start of trusted system instructions.",
    "<|user|>": "Start of a user message.",
    "<|assistant|>": "Start of an assistant/model message.",
    "<|tool_call|>": "Start of a structured tool-call action.",
    "<|tool_result|>": "Start of a structured tool result.",
    "<|terminal|>": "Boundary for terminal-oriented text or output.",
    "<|document|>": "Boundary for retrieved or training document content.",
    "<|code|>": "Boundary for source-code content.",
}

CORE_ARTIFACT_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocabulary.txt",
    "vocab.json",
    "merges.txt",
)


def write_candidate_artifacts(
    backend: Tokenizer,
    config: TokenizerConfig,
    corpus: CorpusPlan,
    *,
    fixture_artifact: bool,
    target_directory: Path | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    target = target_directory or config.candidate_directory
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        assert temporary is not None
        backend.save(str(temporary / "tokenizer.json"), pretty=False)
        tokenizer_payload = json.loads((temporary / "tokenizer.json").read_text(encoding="utf-8"))
        model_payload = tokenizer_payload.get("model", {})
        vocabulary = model_payload.get("vocab")
        merges = model_payload.get("merges")
        if not isinstance(vocabulary, dict) or not isinstance(merges, list):
            raise ValueError("trained tokenizer did not expose BPE vocabulary and merges")

        atomic_write_json(temporary / "tokenizer_config.json", config.resolved_dict())
        special_ids = {token: backend.token_to_id(token) for token in config.special_tokens}
        if any(identifier is None for identifier in special_ids.values()):
            raise ValueError("trained tokenizer is missing a required special token")
        special_payload = {
            "by_token": special_ids,
            "tokens": [
                {"token": token, "id": special_ids[token], "purpose": SPECIAL_TOKEN_PURPOSES[token]}
                for token in config.special_tokens
            ],
            "literal_text_requires_explicit_parsing": True,
        }
        atomic_write_json(temporary / "special_tokens_map.json", special_payload)
        atomic_write_json(temporary / "vocab.json", vocabulary)
        merge_lines = ["#version: 0.2"] + [" ".join(merge) if isinstance(merge, list) else str(merge) for merge in merges]
        atomic_write_text(temporary / "merges.txt", "\n".join(merge_lines) + "\n")
        vocabulary_lines = [
            f"{identifier}\t{json.dumps(token, ensure_ascii=False)}"
            for token, identifier in sorted(vocabulary.items(), key=lambda item: item[1])
        ]
        atomic_write_text(temporary / "vocabulary.txt", "\n".join(vocabulary_lines) + "\n")
        if fixture_artifact:
            atomic_write_text(
                temporary / "FIXTURE_ONLY.txt",
                "NON-PRODUCTION TOKENIZER: trained only to verify the Phase 3 pipeline.\n",
            )

        artifact_hashes = {name: sha256_file(temporary / name) for name in CORE_ARTIFACT_NAMES}
        manifest = {
            "schema_version": 1,
            "tokenizer_algorithm": config.algorithm,
            "requested_vocabulary_size": config.vocabulary_size,
            "actual_vocabulary_size": backend.get_vocab_size(with_added_tokens=True),
            "special_tokens": special_ids,
            "seed": config.seed,
            "package_versions": {
                "python": platform.python_version(),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "cyber-agent": importlib.metadata.version("cyber-agent"),
            },
            "creation_timestamp": utc_now(),
            "input_split": str(config.training_split_path.relative_to(config.project_root)),
            "input_document_count": corpus.document_count,
            "input_character_count": corpus.character_count,
            "input_byte_count": corpus.byte_count,
            "source_counts": corpus.source_counts,
            "category_counts": corpus.category_counts,
            "license_counts": corpus.license_counts,
            "input_manifest_hashes": corpus.input_manifest_hashes,
            "excluded_split_document_counts": corpus.excluded_split_counts,
            "configuration_hash": config.configuration_hash(),
            "tokenizer_artifact_hashes": artifact_hashes,
            "training_documents": corpus.manifest_documents(),
            "fixture_artifact": fixture_artifact,
            "production_ready": False,
            **(manifest_extra or {}),
        }
        atomic_write_json(temporary / "training_manifest.json", manifest)
        _publish_directory(temporary, target)
        temporary = None
        return target, manifest
    finally:
        if temporary is not None and temporary.exists() and temporary.is_dir():
            shutil.rmtree(temporary)


def verify_candidate(candidate_directory: Path) -> dict[str, Any]:
    manifest_path = candidate_directory / "training_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"candidate training manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"candidate training manifest is invalid: {exc}") from exc
    hashes = manifest.get("tokenizer_artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("candidate manifest has no artifact hashes")
    for name, expected_hash in hashes.items():
        path = candidate_directory / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"candidate artifact hash mismatch: {name}")
    return manifest


def export_final_tokenizer(config: TokenizerConfig, candidate_size: int) -> Path:
    candidate = config.candidates_directory / str(candidate_size)
    verify_candidate(candidate)
    if not (candidate / "evaluation_report.json").exists():
        raise ValueError("candidate must be evaluated before final export")
    target = config.final_directory
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=".final.", suffix=".tmp", dir=target.parent))
    try:
        assert temporary is not None
        for source in candidate.iterdir():
            destination = temporary / source.name
            if source.is_file():
                shutil.copy2(source, destination)
        atomic_write_json(
            temporary / "selection_manifest.json",
            {
                "selected_candidate": candidate_size,
                "selected_at": utc_now(),
                "source_training_manifest_hash": sha256_file(candidate / "training_manifest.json"),
                "source_evaluation_report_hash": sha256_file(candidate / "evaluation_report.json"),
                "selection_is_explicit": True,
            },
        )
        _publish_directory(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None and temporary.exists() and temporary.is_dir():
            shutil.rmtree(temporary)


def deterministic_artifact_hashes(candidate_directory: Path) -> dict[str, str]:
    return {name: sha256_file(candidate_directory / name) for name in CORE_ARTIFACT_NAMES}


def _publish_directory(temporary: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if target.exists():
            os.replace(target, backup)
        os.replace(temporary, target)
    except Exception:
        if not target.exists() and backup.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
