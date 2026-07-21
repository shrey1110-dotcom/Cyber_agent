"""Content-free aggregate audit and licensing reports."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.balance import estimate_pre_tokenizer_tokens
from cyber_agent.data_pipeline.export import atomic_write_json, read_jsonl
from cyber_agent.data_pipeline.schemas import utc_now
from cyber_agent.data_pipeline.sources import SourceRegistry


def representative_samples(config: PipelineConfig, *, seed: int, per_category: int = 2) -> dict[str, list[dict[str, Any]]]:
    source_path = config.paths.cleaned / "balanced.jsonl"
    if not source_path.exists():
        source_path = config.paths.cleaned / "deduplicated.jsonl"
    documents = read_jsonl(source_path)
    categories = sorted({str(document["category"]) for document in documents})
    result: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        candidates = sorted(
            (document for document in documents if document["category"] == category),
            key=lambda document: hashlib.sha256(f"{seed}:{document['document_id']}".encode("utf-8")).hexdigest(),
        )
        result[category] = [
            {
                "document_id": document["document_id"],
                "source_name": document["source_name"],
                "license": document["license"],
                "text_preview": document["text"][:500],
            }
            for document in candidates[:per_category]
        ]
    return result


def run_report(config: PipelineConfig) -> dict[str, Any]:
    split_records = {
        name: read_jsonl(config.paths.splits / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    documents = [record for records in split_records.values() for record in records]
    rejections = read_jsonl(config.paths.manifests / "rejection_manifest.jsonl")
    duplicates = read_jsonl(config.paths.reports / "duplicate_report.jsonl")
    registry = SourceRegistry.load(config.paths)

    by_license = Counter(document["license"] for document in documents)
    by_source = Counter(document["source_name"] for document in documents)
    by_category = Counter(document["category"] for document in documents)
    by_rejection_reason = Counter(
        code
        for rejection in rejections
        for code in rejection.get("reason_codes", ["unspecified"])
    )
    by_rejection_stage = Counter(rejection["stage"] for rejection in rejections)
    duplicate_types = Counter(record["duplicate_type"] for record in duplicates)
    unresolved = [
        {
            "source_name": source.source_name,
            "license": source.license,
            "reason": source.notes or source.allowed_use,
        }
        for source in registry.all_sources()
        if not source.enabled
        or not source.is_approved
        or config.license_policy.rule_for(source.license) is None
        or config.license_policy.rule_for(source.license).status != "allowed"  # type: ignore[union-attr]
    ]
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "accepted_documents": len(documents),
        "raw_documents": len(read_jsonl(config.paths.raw / "documents.jsonl")),
        "estimated_pre_tokenizer_tokens": sum(estimate_pre_tokenizer_tokens(document["text"]) for document in documents),
        "total_characters": sum(len(document["text"]) for document in documents),
        "average_quality_score": round(
            sum(float(document["quality_score"]) for document in documents) / max(1, len(documents)),
            6,
        ),
        "by_split": {name: len(records) for name, records in split_records.items()},
        "by_license": dict(sorted(by_license.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_category": dict(sorted(by_category.items())),
        "rejections": {
            "total": len(rejections),
            "by_reason_code": dict(sorted(by_rejection_reason.items())),
            "by_stage": dict(sorted(by_rejection_stage.items())),
        },
        "duplicates": {
            "removed": len(duplicates),
            "exact": duplicate_types["exact"],
            "near": duplicate_types["near"],
            "groups": len({record["duplicate_group_id"] for record in duplicates}),
        },
        "licensing": {
            "policy_version": config.license_policy.policy_version,
            "unresolved_source_assumptions": unresolved,
        },
        "dataset_mode": config.dataset_mode.to_dict(),
        "sources_requiring_review_before_open_weight_release": sorted(
            source.source_name for source in registry.all_sources()
            if source.source_name in by_source and (
                source.license == "REVIEW_REQUIRED"
                or config.license_policy.rule_for(source.license) is None
                or config.license_policy.rule_for(source.license).status != "allowed"  # type: ignore[union-attr]
                or not config.dataset_mode.release_cleared
            )
        ),
        "quality_scoring": {
            "minimum_score": config.minimum_quality_score,
            "method": "Equal-weight length, token diversity, readability, English-marker, and structure components after hard rejection checks.",
        },
    }
    atomic_write_json(config.paths.reports / "dataset_summary.json", summary)
    atomic_write_json(
        config.paths.reports / "license_counts.json",
        {"policy_version": config.license_policy.policy_version, "document_counts": dict(sorted(by_license.items()))},
    )
    return summary
