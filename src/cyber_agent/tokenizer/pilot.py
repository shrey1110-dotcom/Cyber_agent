"""Frozen-snapshot tokenizer candidates, evaluation, comparison, and export."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cyber_agent.data_pipeline.export import atomic_write_json, atomic_write_text, iter_jsonl, read_jsonl
from cyber_agent.data_pipeline.schemas import Document, utc_now
from cyber_agent.data_pipeline.snapshot import verify_snapshot
from cyber_agent.tokenizer.artifacts import verify_candidate, write_candidate_artifacts
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.corpus import CorpusDocumentReference, sha256_file
from cyber_agent.tokenizer.evaluator import EvaluationExample
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.tokenizer.model_budget import estimate_model_budget


def _snapshot_document(value: dict[str, Any]) -> Document:
    return Document.from_dict({key: value[key] for key in (
        "document_id", "text", "source_name", "source_url", "license", "category",
        "language", "retrieved_at", "content_hash", "quality_score", "metadata",
    )})


@dataclass(frozen=True, slots=True)
class FrozenCorpusPlan:
    config: TokenizerConfig
    snapshot_directory: Path
    snapshot_manifest: dict[str, Any]
    references: tuple[CorpusDocumentReference, ...]
    input_manifest_hashes: dict[str, str]
    excluded_split_counts: dict[str, int]

    @property
    def document_count(self) -> int:
        return len(self.references)

    @property
    def character_count(self) -> int:
        return sum(reference.character_count for reference in self.references)

    @property
    def byte_count(self) -> int:
        return sum(reference.byte_count for reference in self.references)

    @property
    def source_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reference.source_name for reference in self.references).items()))

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reference.category for reference in self.references).items()))

    @property
    def license_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reference.license for reference in self.references).items()))

    def iter_texts(self) -> Iterator[str]:
        selected = {reference.document_id for reference in self.references}
        yielded: set[str] = set()
        for value in iter_jsonl(self.snapshot_directory / "train_manifest.jsonl"):
            document = _snapshot_document(value)
            if document.document_id in selected:
                yielded.add(document.document_id)
                yield document.text
        if yielded != selected:
            raise ValueError("frozen training documents changed while tokenizer training was running")

    def manifest_documents(self) -> list[dict[str, Any]]:
        return [reference.to_manifest_dict() for reference in sorted(self.references, key=lambda item: item.document_id)]


def load_frozen_corpus(config: TokenizerConfig, snapshot_name: str) -> FrozenCorpusPlan:
    snapshot_directory = config.project_root / "artifacts" / "datasets" / "snapshots" / snapshot_name
    manifest = verify_snapshot(snapshot_directory)
    train_path = snapshot_directory / "train_manifest.jsonl"
    validation_path = snapshot_directory / "validation_manifest.jsonl"
    test_path = snapshot_directory / "test_manifest.jsonl"
    train_records: list[dict[str, Any]] = []
    validation_ids: set[str] = set(); test_ids: set[str] = set()
    for record in iter_jsonl(train_path): train_records.append(record)
    for record in iter_jsonl(validation_path): validation_ids.add(str(record.get("document_id")))
    for record in iter_jsonl(test_path): test_ids.add(str(record.get("document_id")))
    train_ids = {str(record.get("document_id")) for record in train_records}
    heldout_ids = validation_ids | test_ids
    if train_ids & heldout_ids:
        raise ValueError("frozen tokenizer training split overlaps validation or test")
    references: list[CorpusDocumentReference] = []
    for line_number, value in enumerate(train_records, start=1):
        if value.get("split") != "train":
            raise ValueError("frozen training manifest contains a non-training record")
        document = _snapshot_document(value)
        references.append(
            CorpusDocumentReference(
                document.document_id,
                document.content_hash,
                document.source_name,
                document.category,
                document.license,
                line_number,
                len(document.text),
                len(document.text.encode("utf-8")),
                line_number,
            )
        )
    if not references:
        raise ValueError("frozen snapshot has no training documents")
    # Release the parsed JSON objects before tokenizer training; references
    # retain only compact provenance fields.
    train_records.clear()
    hashes = {
        filename: sha256_file(snapshot_directory / filename)
        for filename in (
            "snapshot_manifest.json", "source_manifest.jsonl", "license_manifest.jsonl",
            "train_manifest.jsonl", "validation_manifest.jsonl", "test_manifest.jsonl",
        )
    }
    return FrozenCorpusPlan(
        config,
        snapshot_directory,
        manifest,
        tuple(references),
        hashes,
            {"validation": len(validation_ids), "test": len(test_ids)},
    )


def train_snapshot_candidates(
    config: TokenizerConfig,
    *,
    snapshot_name: str,
    vocabulary_sizes: Iterable[int] | None = None,
    fixture_artifact: bool | None = None,
) -> dict[str, Any]:
    plan = load_frozen_corpus(config, snapshot_name)
    sizes = tuple(vocabulary_sizes or config.candidate_vocabulary_sizes)
    if not sizes or any(size < 300 for size in sizes):
        raise ValueError("candidate vocabulary sizes must be at least 300")
    fixture = plan.document_count < 100 if fixture_artifact is None else fixture_artifact
    results: list[dict[str, Any]] = []
    expected_special_ids = {token: index for index, token in enumerate(config.special_tokens)}
    for size in sizes:
        selected = config.with_overrides(vocabulary_size=size)
        backend = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
        backend.normalizer = None
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=size,
            min_frequency=selected.minimum_frequency,
            show_progress=False,
            special_tokens=list(selected.special_tokens),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=selected.maximum_token_length,
        )
        backend.train_from_iterator(plan.iter_texts(), trainer=trainer, length=plan.document_count)
        special_ids = {token: backend.token_to_id(token) for token in selected.special_tokens}
        if special_ids != expected_special_ids:
            raise ValueError("candidate special-token IDs differ from the stable contract")
        target = selected.candidates_directory / snapshot_name / str(size)
        directory, manifest = write_candidate_artifacts(
            backend,
            selected,
            plan,  # type: ignore[arg-type]
            fixture_artifact=fixture,
            target_directory=target,
            manifest_extra={
                "snapshot_name": snapshot_name,
                "snapshot_content_hash": plan.snapshot_manifest["snapshot_content_hash"],
                "frozen_train_manifest_hash": plan.input_manifest_hashes["train_manifest.jsonl"],
                "train_only_verified": True,
                "requested_size_was_fully_produced": backend.get_vocab_size(with_added_tokens=True) == size,
                "production_ready": False,
                "local_research_only": bool(plan.snapshot_manifest.get("local_research_only")),
                "release_cleared": bool(plan.snapshot_manifest.get("release_cleared")),
                "weight_publication_allowed": bool(plan.snapshot_manifest.get("weight_publication_allowed")),
            },
        )
        results.append(
            {
                "requested_vocabulary_size": size,
                "actual_vocabulary_size": manifest["actual_vocabulary_size"],
                "requested_size_was_fully_produced": manifest["requested_size_was_fully_produced"],
                "candidate_directory": str(directory),
                "artifact_hashes": manifest["tokenizer_artifact_hashes"],
            }
        )
    return {
        "status": "complete",
        "snapshot": snapshot_name,
        "snapshot_content_hash": plan.snapshot_manifest["snapshot_content_hash"],
        "fixture_artifact": fixture,
        "candidates": results,
    }


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(row["tokens"] for row in rows)
    total_characters = sum(row["characters"] for row in rows)
    total_bytes = sum(row["bytes"] for row in rows)
    lengths = [row["tokens"] for row in rows]
    return {
        "documents": len(rows),
        "total_tokens": total_tokens,
        "total_characters": total_characters,
        "total_bytes": total_bytes,
        "characters_per_token": round(total_characters / max(1, total_tokens), 6),
        "bytes_per_token": round(total_bytes / max(1, total_tokens), 6),
        "tokens_per_document": round(total_tokens / max(1, len(rows)), 6),
        "longest_tokenized_sequence": max(lengths, default=0),
        "sequence_length_percentiles": {
            "p50": _percentile(lengths, 0.50), "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95), "p99": _percentile(lengths, 0.99),
        },
        "unknown_token_rate": round(sum(row["unknown_tokens"] for row in rows) / max(1, total_tokens), 12),
        "round_trip_accuracy": round(sum(row["round_trip"] for row in rows) / max(1, len(rows)), 12),
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(values) for name, values in sorted(grouped.items())}


def _fragmentation(tokenizer: CyberTokenizer) -> dict[str, Any]:
    groups = {
        "common_security_terms": ["authentication", "vulnerability", "ransomware", "least privilege", "zero trust"],
        "commands_and_flags": ["journalctl -u ssh", "grep -R", "--since", "chmod 600", "curl --fail"],
        "technical_identifiers": ["CVE-2026-12345", "CWE-79", "T1059.004", "TA0001"],
    }
    return {
        group: {
            value: {"token_count": len(tokenizer.encode(value)), "tokens": tokenizer.tokens_for_ids(tokenizer.encode(value))}
            for value in values
        }
        for group, values in groups.items()
    }


def evaluate_snapshot_candidate(config: TokenizerConfig, *, snapshot_name: str, candidate_size: int) -> dict[str, Any]:
    plan = load_frozen_corpus(config, snapshot_name)
    directory = config.candidates_directory / snapshot_name / str(candidate_size)
    training_manifest = verify_candidate(directory)
    if training_manifest.get("snapshot_content_hash") != plan.snapshot_manifest["snapshot_content_hash"]:
        raise ValueError("candidate was not trained from the requested frozen snapshot")
    tokenizer = CyberTokenizer.from_file(directory / "tokenizer.json")
    fixture_path = config.project_root / "fixtures" / "tokenizer_evaluation.jsonl"
    examples = [
        EvaluationExample(
            str(value["name"]), str(value["evaluation_group"]), str(value["category"]),
            str(value["source_name"]), str(value["text"]),
        )
        for value in read_jsonl(fixture_path)
    ]
    evaluation_hashes = {"representative_fixture": sha256_file(fixture_path)}
    for split_name in ("validation", "test"):
        path = plan.snapshot_directory / f"{split_name}_manifest.jsonl"
        evaluation_hashes[f"frozen_{split_name}_manifest"] = sha256_file(path)
        for value in read_jsonl(path):
            document = _snapshot_document(value)
            examples.append(EvaluationExample(document.document_id, f"frozen_{split_name}", document.category, document.source_name, document.text))
    rows: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    for example in examples:
        identifiers = tokenizer.encode(example.text)
        decoded = tokenizer.decode(identifiers)
        language = {
            "python": "python", "bash": "shell", "json": "json", "yaml": "yaml"
        }.get(example.evaluation_group, "not_applicable")
        row = {
            "name": example.name, "group": example.evaluation_group, "category": example.category,
            "source": example.source_name, "programming_language": language,
            "characters": len(example.text), "bytes": len(example.text.encode("utf-8")), "tokens": len(identifiers),
            "unknown_tokens": sum(identifier == tokenizer.unk_token_id for identifier in identifiers),
            "round_trip": decoded == example.text,
        }
        rows.append(row)
        if len(inspection_rows) < 20:
            inspection_rows.append({
                "name": example.name, "text": example.text[:500], "token_ids": identifiers[:200],
                "tokens": tokenizer.tokens_for_ids(identifiers[:200]), "truncated": len(identifiers) > 200,
            })
    train_token_count = 0
    train_unknown_count = 0
    for text in plan.iter_texts():
        identifiers = tokenizer.encode(text)
        train_token_count += len(identifiers)
        train_unknown_count += sum(identifier == tokenizer.unk_token_id for identifier in identifiers)
    special_behavior = {}
    for token, identifier in tokenizer.special_token_ids.items():
        literal = tokenizer.encode(token)
        special_behavior[token] = {
            "id": identifier,
            "literal_is_ordinary_content": literal != [identifier] and tokenizer.decode(literal) == token,
            "trusted_parse_id": tokenizer.encode(token, parse_special_tokens=True),
        }
    actual_vocab = tokenizer.vocabulary_size
    report = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "snapshot_name": snapshot_name,
        "snapshot_content_hash": plan.snapshot_manifest["snapshot_content_hash"],
        "requested_vocabulary_size": training_manifest["requested_vocabulary_size"],
        "actual_vocabulary_size": actual_vocab,
        "requested_size_was_fully_produced": actual_vocab == training_manifest["requested_vocabulary_size"],
        "fixture_artifact": bool(training_manifest.get("fixture_artifact")),
        "local_research_only": bool(plan.snapshot_manifest.get("local_research_only")),
        "release_cleared": bool(plan.snapshot_manifest.get("release_cleared")),
        "production_ready": False,
        "weight_publication_allowed": bool(plan.snapshot_manifest.get("weight_publication_allowed")),
        "evaluation_document_count": len(rows),
        "training_documents_used_for_evaluation": 0,
        "evaluation_input_hashes": evaluation_hashes,
        "estimated_pre_tokenizer_tokens": plan.snapshot_manifest["estimated_token_count"],
        "exact_candidate_token_counts": {"training": train_token_count, "evaluation": sum(row["tokens"] for row in rows)},
        "overall": _aggregate(rows),
        "by_category": _group(rows, "category"),
        "by_source": _group(rows, "source"),
        "by_programming_language": _group(rows, "programming_language"),
        "by_evaluation_group": _group(rows, "group"),
        "longest_sequences": sorted(
            ({"name": row["name"], "tokens": row["tokens"]} for row in rows),
            key=lambda row: (-row["tokens"], row["name"]),
        )[:20],
        "round_trip_failures": [row["name"] for row in rows if not row["round_trip"]],
        "special_token_behavior": special_behavior,
        "zero_unknown_token_dependence": all(row["unknown_tokens"] == 0 for row in rows),
        "fragmentation": _fragmentation(tokenizer),
        "human_readable_token_inspections": inspection_rows,
        "training_corpus_coverage": {
            "documents": plan.document_count,
            "tokens": train_token_count,
            "unknown_token_rate": round(train_unknown_count / max(1, train_token_count), 12),
        },
        "evaluation_corpus_coverage": _aggregate(rows),
        "model_budget": estimate_model_budget((actual_vocab,)),
        "selection_warning": "Fixture evidence is mechanics-only and cannot justify a production tokenizer."
        if training_manifest.get("fixture_artifact") else None,
    }
    atomic_write_json(directory / "evaluation_report.json", report)
    return report


def _mean_fragmentation(report: dict[str, Any]) -> float:
    values = [
        details["token_count"]
        for group in report["fragmentation"].values()
        for details in group.values()
    ]
    return sum(values) / max(1, len(values))


def compare_snapshot_candidates(
    config: TokenizerConfig,
    *,
    snapshot_name: str,
    minimum_evaluation_documents: int = 1000,
    minimum_training_estimated_tokens: int = 10_000_000,
    hidden_size: int = 512,
) -> dict[str, Any]:
    plan = load_frozen_corpus(config, snapshot_name)
    reviewed_sources = read_jsonl(plan.snapshot_directory / "source_manifest.jsonl")
    root = config.candidates_directory / snapshot_name
    candidates: list[dict[str, Any]] = []
    training_manifests: list[dict[str, Any]] = []
    for size in config.candidate_vocabulary_sizes:
        report_path = root / str(size) / "evaluation_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        training_manifest = verify_candidate(root / str(size))
        training_manifests.append(training_manifest)
        candidates.append({
            "requested_vocabulary_size": size,
            "actual_vocabulary_size": report["actual_vocabulary_size"],
            "characters_per_token": report["overall"]["characters_per_token"],
            "bytes_per_token": report["overall"]["bytes_per_token"],
            "unknown_token_rate": report["overall"]["unknown_token_rate"],
            "round_trip_accuracy": report["overall"]["round_trip_accuracy"],
            "mean_fragmentation": round(_mean_fragmentation(report), 6),
            "evaluation_document_count": report["evaluation_document_count"],
            "estimated_pre_tokenizer_tokens": report["estimated_pre_tokenizer_tokens"],
            "exact_candidate_token_counts": report["exact_candidate_token_counts"],
            "fixture_artifact": report["fixture_artifact"],
            "snapshot_content_hash": training_manifest.get("snapshot_content_hash"),
            "frozen_train_manifest_hash": training_manifest.get("frozen_train_manifest_hash"),
            "special_token_ids": training_manifest.get("special_tokens"),
            "vocabulary_parameter_cost": estimate_model_budget((report["actual_vocabulary_size"],), hidden_size=hidden_size),
        })
    missing = sorted(set(config.candidate_vocabulary_sizes) - {row["requested_vocabulary_size"] for row in candidates})
    special_contracts = {
        json.dumps(manifest.get("special_tokens"), sort_keys=True)
        for manifest in training_manifests
    }
    training_hashes = {
        manifest.get("frozen_train_manifest_hash")
        for manifest in training_manifests
    }
    snapshot_hashes = {
        manifest.get("snapshot_content_hash")
        for manifest in training_manifests
    }
    literal_special_tokens_safe = bool(candidates) and all(
        all(details.get("literal_is_ordinary_content") for details in report["special_token_behavior"].values())
        for report in (
            json.loads((root / str(row["requested_vocabulary_size"]) / "evaluation_report.json").read_text(encoding="utf-8"))
            for row in candidates
        )
    )
    evidence = {
        "all_candidates_present": not missing,
        "minimum_evaluation_documents_met": bool(candidates) and min(row["evaluation_document_count"] for row in candidates) >= minimum_evaluation_documents,
        "minimum_training_tokens_met": bool(candidates) and min(row["estimated_pre_tokenizer_tokens"] for row in candidates) >= minimum_training_estimated_tokens,
        "round_trip_correct": bool(candidates) and all(row["round_trip_accuracy"] == 1.0 for row in candidates),
        "zero_unknown_rate": bool(candidates) and all(row["unknown_token_rate"] == 0.0 for row in candidates),
        "non_fixture_corpus": bool(candidates) and not any(row["fixture_artifact"] for row in candidates),
        "same_frozen_training_snapshot": bool(training_manifests)
        and len(training_hashes) == 1
        and len(snapshot_hashes) == 1
        and next(iter(snapshot_hashes)) == plan.snapshot_manifest["snapshot_content_hash"],
        "stable_special_token_ids": bool(training_manifests) and len(special_contracts) == 1,
        "literal_control_tokens_are_untrusted_content": literal_special_tokens_safe,
        "all_sources_approved_for_production": bool(reviewed_sources)
        and all(source.get("review_status") == "approved_for_production" for source in reviewed_sources),
    }
    local_research_only = bool(plan.snapshot_manifest.get("local_research_only"))
    release_evidence_name = "all_sources_approved_for_production"
    core_evidence = {
        name: value for name, value in evidence.items()
        if name != release_evidence_name
    }
    eligible = all(core_evidence.values()) and (local_research_only or evidence[release_evidence_name])
    recommendation = None
    if eligible:
        recommendation = min(
            candidates,
            key=lambda row: (
                row["mean_fragmentation"],
                -row["characters_per_token"],
                row["actual_vocabulary_size"],
            ),
        )["requested_vocabulary_size"]
    report = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "snapshot_name": snapshot_name,
        "candidates": candidates,
        "missing_candidates": missing,
        "minimum_evidence_thresholds": {
            "evaluation_documents": minimum_evaluation_documents,
            "estimated_training_tokens": minimum_training_estimated_tokens,
        },
        "evidence": evidence,
        "comparison_status": (
            "local_research_recommendation_ready" if eligible and local_research_only
            else "recommendation_ready" if eligible
            else "insufficient_evidence"
        ),
        "recommended_candidate": recommendation,
        "selection_considers": [
            "compression", "code and command fragmentation", "technical identifier fragmentation",
            "vocabulary parameter cost", "corpus size", "candidate stability", "round-trip correctness",
            "special-token behavior", "intended approximately 50M-parameter architecture",
        ],
        "default_export_status": "pilot_only",
        "local_research_only": local_research_only,
        "release_cleared": bool(plan.snapshot_manifest.get("release_cleared")),
        "weight_publication_allowed": bool(plan.snapshot_manifest.get("weight_publication_allowed")),
        "publication_warning": "A local tokenizer recommendation does not clear dataset or model-weight publication."
        if local_research_only else None,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "selection_report.json", report)
    atomic_write_json(config.output_directory / "comparison_report.json", report)
    return report


def export_snapshot_candidate(
    config: TokenizerConfig,
    *,
    snapshot_name: str,
    candidate_size: int,
    confirm: bool,
    status: str = "pilot_only",
) -> Path:
    if not confirm:
        raise ValueError("final export requires explicit --confirm")
    if status not in {"pilot_only", "production_candidate", "production_frozen"}:
        raise ValueError("unsupported tokenizer export status")
    source = config.candidates_directory / snapshot_name / str(candidate_size)
    manifest = verify_candidate(source)
    if manifest.get("snapshot_name") != snapshot_name:
        raise ValueError("candidate snapshot provenance does not match")
    required_source = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "training_manifest.json",
        "evaluation_report.json", "vocabulary.txt",
    }
    missing = sorted(name for name in required_source if not (source / name).exists())
    selection_report = config.candidates_directory / snapshot_name / "selection_report.json"
    if missing or not selection_report.exists():
        raise ValueError(f"candidate evaluation or selection artifacts are incomplete: {', '.join(missing)}")
    selection = json.loads(selection_report.read_text(encoding="utf-8"))
    snapshot_manifest = verify_snapshot(config.project_root / "artifacts" / "datasets" / "snapshots" / snapshot_name)
    if status != "pilot_only" and not snapshot_manifest.get("weight_publication_allowed", False):
        raise ValueError("snapshot is not cleared for public model-weight or production tokenizer export")
    if status != "pilot_only":
        if selection.get("comparison_status") != "recommendation_ready" or selection.get("recommended_candidate") != candidate_size:
            raise ValueError("production status requires a threshold-qualified selection recommendation")
    target = config.final_directory / snapshot_name
    if target.exists():
        raise ValueError("final tokenizer export already exists and will not be overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{snapshot_name}.", suffix=".tmp", dir=target.parent))
    try:
        assert temporary is not None
        for name in sorted(required_source):
            shutil.copy2(source / name, temporary / name)
        shutil.copy2(selection_report, temporary / "selection_report.json")
        atomic_write_json(
            temporary / "export_manifest.json",
            {
                "schema_version": 1, "snapshot_name": snapshot_name, "candidate": candidate_size,
                "status": status, "exported_at": utc_now(), "explicit_confirmation": True,
                "snapshot_content_hash": manifest["snapshot_content_hash"],
                "local_research_only": bool(snapshot_manifest.get("local_research_only")),
                "release_cleared": bool(snapshot_manifest.get("release_cleared")),
                "weight_publication_allowed": bool(snapshot_manifest.get("weight_publication_allowed")),
            },
        )
        checksum_names = sorted((*required_source, "selection_report.json", "export_manifest.json"))
        atomic_write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checksum_names),
        )
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
