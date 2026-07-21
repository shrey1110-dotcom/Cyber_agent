"""Representative tokenizer metrics, candidate comparison, and tradeoff guidance."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cyber_agent.data_pipeline.export import atomic_write_json, read_jsonl
from cyber_agent.data_pipeline.schemas import Document, utc_now
from cyber_agent.tokenizer.artifacts import verify_candidate
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.corpus import sha256_file
from cyber_agent.tokenizer.loader import CyberTokenizer


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    name: str
    evaluation_group: str
    category: str
    source_name: str
    text: str


def load_evaluation_examples(config: TokenizerConfig) -> tuple[list[EvaluationExample], dict[str, str]]:
    fixture_path = config.project_root / "fixtures" / "tokenizer_evaluation.jsonl"
    if not fixture_path.exists():
        raise ValueError(f"tokenizer evaluation fixture is missing: {fixture_path}")
    examples: list[EvaluationExample] = []
    for value in read_jsonl(fixture_path):
        try:
            example = EvaluationExample(
                name=value["name"],
                evaluation_group=value["evaluation_group"],
                category=value["category"],
                source_name=value["source_name"],
                text=value["text"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid tokenizer evaluation fixture record: {exc}") from exc
        if not example.text:
            raise ValueError(f"empty tokenizer evaluation example: {example.name}")
        examples.append(example)

    input_hashes = {"representative_fixture": sha256_file(fixture_path)}
    for split_name in ("validation", "test"):
        split_path = config.project_root / "data" / "splits" / f"{split_name}.jsonl"
        if not split_path.exists():
            raise ValueError(f"Phase 2 {split_name} split is missing")
        input_hashes[f"phase2_{split_name}_split"] = sha256_file(split_path)
        for value in read_jsonl(split_path):
            document = Document.from_dict(value)
            examples.append(
                EvaluationExample(
                    name=document.document_id,
                    evaluation_group=f"phase2_{split_name}",
                    category=document.category,
                    source_name=document.source_name,
                    text=document.text,
                )
            )
    return examples, input_hashes


def evaluate_tokenizer(config: TokenizerConfig, tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_path = tokenizer_path.resolve()
    candidate_directory = tokenizer_path.parent
    training_manifest = verify_candidate(candidate_directory)
    tokenizer = CyberTokenizer.from_file(tokenizer_path)
    examples, evaluation_input_hashes = load_evaluation_examples(config)

    rows: list[dict[str, Any]] = []
    for example in examples:
        token_ids = tokenizer.encode(example.text)
        decoded = tokenizer.decode(token_ids)
        rows.append(
            {
                "name": example.name,
                "group": example.evaluation_group,
                "category": example.category,
                "source": example.source_name,
                "characters": len(example.text),
                "bytes": len(example.text.encode("utf-8")),
                "tokens": len(token_ids),
                "unknown_tokens": sum(identifier == tokenizer.unk_token_id for identifier in token_ids),
                "round_trip": decoded == example.text,
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "tokenizer_path": str(tokenizer_path.relative_to(config.project_root)),
        "requested_vocabulary_size": training_manifest["requested_vocabulary_size"],
        "actual_vocabulary_size": tokenizer.vocabulary_size,
        "fixture_artifact": bool(training_manifest.get("fixture_artifact")),
        "evaluation_document_count": len(rows),
        "phase2_training_documents_used_for_evaluation": 0,
        "evaluation_input_hashes": evaluation_input_hashes,
        "overall": _aggregate(rows),
        "by_evaluation_group": _group_metrics(rows, "group"),
        "by_category": _group_metrics(rows, "category"),
        "by_source": _group_metrics(rows, "source"),
        "special_token_behavior": _special_token_metrics(tokenizer),
        "round_trip_failures": [row["name"] for row in rows if not row["round_trip"]],
        "selection_warning": "Fixture metrics validate mechanics only and must not be used to select a production vocabulary."
        if training_manifest.get("fixture_artifact")
        else None,
    }
    atomic_write_json(candidate_directory / "evaluation_report.json", report)
    return report


def compare_candidates(config: TokenizerConfig) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for directory in sorted(config.candidates_directory.iterdir(), key=lambda path: path.name):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        report_path = directory / "evaluation_report.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation report for candidate {directory.name}: {exc}") from exc
        overall = report["overall"]
        candidates.append(
            {
                "requested_vocabulary_size": report["requested_vocabulary_size"],
                "actual_vocabulary_size": report["actual_vocabulary_size"],
                "evaluation_document_count": report["evaluation_document_count"],
                "tokens": overall["total_tokens"],
                "characters_per_token": overall["characters_per_token"],
                "bytes_per_token": overall["bytes_per_token"],
                "unknown_token_rate": overall["unknown_token_rate"],
                "round_trip_accuracy": overall["round_trip_accuracy"],
                "fixture_artifact": report.get("fixture_artifact", False),
            }
        )
    if not candidates:
        raise ValueError("no evaluated tokenizer candidates were found")
    standard_sizes = set(config.candidate_vocabulary_sizes)
    present_sizes = {candidate["requested_vocabulary_size"] for candidate in candidates}
    enough_documents = min(candidate["evaluation_document_count"] for candidate in candidates) >= 1000
    comparison_valid = standard_sizes <= present_sizes and enough_documents and not any(
        candidate["fixture_artifact"] for candidate in candidates
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "candidates": sorted(candidates, key=lambda candidate: candidate["requested_vocabulary_size"]),
        "standard_candidate_sizes": list(config.candidate_vocabulary_sizes),
        "comparison_status": "measurement_ready" if comparison_valid else "insufficient_test_data",
        "missing_standard_candidates": sorted(standard_sizes - present_sizes),
        "automatic_selection": None,
        "guidance": [
            "Smaller vocabularies use more tokens per sequence but reserve fewer model parameters for embeddings and output logits.",
            "Larger vocabularies may compress text better but consume more of the approximately 50M-parameter budget.",
            "Embedding parameters scale approximately as vocabulary_size × hidden_size; untied output weights can add the same cost again.",
            "The 24K default is a starting hypothesis, not an automatic winner; select only after representative measurements.",
        ],
    }
    atomic_write_json(config.output_directory / "comparison_report.json", report)
    return report


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    documents = len(records)
    total_tokens = sum(record["tokens"] for record in records)
    total_characters = sum(record["characters"] for record in records)
    total_bytes = sum(record["bytes"] for record in records)
    total_unknown = sum(record["unknown_tokens"] for record in records)
    round_trips = sum(record["round_trip"] for record in records)
    return {
        "documents": documents,
        "total_tokens": total_tokens,
        "total_characters": total_characters,
        "total_bytes": total_bytes,
        "characters_per_token": round(total_characters / max(1, total_tokens), 6),
        "bytes_per_token": round(total_bytes / max(1, total_tokens), 6),
        "tokens_per_document": round(total_tokens / max(1, documents), 6),
        "longest_tokenized_sequence": max((record["tokens"] for record in records), default=0),
        "unknown_token_rate": round(total_unknown / max(1, total_tokens), 9),
        "round_trip_accuracy": round(round_trips / max(1, documents), 9),
    }


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return {name: _aggregate(records) for name, records in sorted(grouped.items())}


def _special_token_metrics(tokenizer: CyberTokenizer) -> dict[str, Any]:
    literal_safety: dict[str, bool] = {}
    explicit_parsing: dict[str, bool] = {}
    for token, identifier in tokenizer.special_token_ids.items():
        literal_ids = tokenizer.encode(token)
        literal_safety[token] = literal_ids != [identifier] and tokenizer.decode(literal_ids) == token
        explicit_parsing[token] = tokenizer.encode(token, parse_special_tokens=True) == [identifier]
    return {
        "ids": tokenizer.special_token_ids,
        "literal_text_safe": literal_safety,
        "explicit_parsing": explicit_parsing,
        "all_literal_text_safe": all(literal_safety.values()),
        "all_explicit_parsing_correct": all(explicit_parsing.values()),
    }

