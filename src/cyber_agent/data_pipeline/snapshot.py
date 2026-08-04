"""Immutable pilot dataset snapshot creation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from cyber_agent.data_pipeline.balance import balance_documents
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    iter_jsonl,
    read_jsonl,
)
from cyber_agent.data_pipeline.schemas import Document, canonical_json, utc_now
from cyber_agent.data_pipeline.sources import SourceDefinition, SourceRegistry
from cyber_agent.data_pipeline.split import split_documents
from cyber_agent.tokenizer.corpus import sha256_file


SNAPSHOT_FILES = (
    "snapshot_manifest.json",
    "source_manifest.jsonl",
    "license_manifest.jsonl",
    "train_manifest.jsonl",
    "validation_manifest.jsonl",
    "test_manifest.jsonl",
    "rejection_summary.json",
    "duplicate_summary.json",
    "balance_report.json",
    "dataset_summary.json",
    "LOCAL_RESEARCH_ONLY.txt",
    "checksums.sha256",
)


def snapshot_directory_name(name: str, version: int) -> str:
    return name if version == 1 else f"{name}.v{version}"


def _source_record(source: SourceDefinition, document_count: int) -> dict[str, Any]:
    value = asdict(source)
    value["content_categories"] = list(source.content_categories)
    value["known_risks"] = list(source.known_risks)
    value["document_count"] = document_count
    return value


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _summarize_rejections(paths: Iterable[Path]) -> dict[str, Any]:
    records = [record for path in paths if path.exists() for record in read_jsonl(path)]
    by_stage = Counter(str(record.get("stage", "unknown")) for record in records)
    by_reason = Counter(
        str(code) for record in records for code in record.get("reason_codes", ["unspecified"])
    )
    sensitive_codes = {
        "aws_access_key", "private_key", "generic_api_token", "github_token", "jwt", "password_assignment"
    }
    return {
        "total": len(records),
        "by_stage": dict(sorted(by_stage.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "sensitive_data_detections": sum(count for code, count in by_reason.items() if code in sensitive_codes),
    }


def freeze_snapshot(
    config: PipelineConfig,
    *,
    name: str,
    seed: int | None = None,
    version: int = 1,
    known_limitations: list[str] | None = None,
) -> dict[str, Any]:
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in name):
        raise ValueError("snapshot name must use only letters, digits, dot, underscore, and hyphen")
    if version < 1:
        raise ValueError("snapshot version must be positive")
    selected_seed = config.default_seed if seed is None else seed
    directory_name = snapshot_directory_name(name, version)
    target = config.paths.snapshots / directory_name
    if target.exists():
        raise ValueError(f"frozen snapshot already exists and cannot be overwritten: {directory_name}")

    deduplicated_path = config.paths.cleaned / "deduplicated.jsonl"
    if not deduplicated_path.exists():
        raise ValueError("deduplicated Phase 2 input is missing; run cleaning and deduplication first")
    documents = [Document.from_dict(value) for value in read_jsonl(deduplicated_path)]
    balance = balance_documents(documents, budget=config.pilot_budget, seed=selected_seed)
    if not balance.selected:
        raise ValueError("pilot budget excluded every cleaned document")
    split_map, assignments = split_documents(
        list(balance.selected),
        seed=selected_seed,
        proportions=config.split_proportions,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{directory_name}.", suffix=".tmp", dir=target.parent))
    try:
        assert temporary is not None
        registry = SourceRegistry.load(config.paths)
        selected_by_source = Counter(document.source_name for document in balance.selected)
        source_records = [
            _source_record(source, selected_by_source[source.source_name])
            for source in sorted(registry.all_sources(), key=lambda item: item.source_name)
            if source.source_name in selected_by_source
        ]
        license_records = []
        for identifier, count in sorted(Counter(document.license for document in balance.selected).items()):
            rule = config.license_policy.require_usable(
                identifier,
                local_research_only=config.dataset_mode.local_research_only,
            )
            evidence = sorted(
                {
                    source.license_evidence_url
                    for source in registry.all_sources()
                    if source.source_name in selected_by_source and source.license == identifier
                }
            )
            license_records.append(
                {
                    "license": identifier,
                    "status": rule.status,
                    "attribution_required": rule.attribution_required,
                    "document_count": count,
                    "evidence_urls": evidence,
                }
            )
        atomic_write_jsonl(temporary / "source_manifest.jsonl", source_records)
        atomic_write_jsonl(temporary / "license_manifest.jsonl", license_records)
        assignment_by_id = {assignment.document_id: assignment for assignment in assignments}
        for split_name in ("train", "validation", "test"):
            records = []
            for document in sorted(split_map[split_name], key=lambda item: item.document_id):
                record = document.to_dict()
                assignment = assignment_by_id[document.document_id]
                record.update({"split": split_name, "duplicate_group_id": assignment.duplicate_group_id})
                records.append(record)
            atomic_write_jsonl(temporary / f"{split_name}_manifest.jsonl", records)

        rejection_summary = _summarize_rejections(
            (config.paths.rejected / "ingest.jsonl", config.paths.rejected / "clean.jsonl")
        )
        duplicate_records = read_jsonl(config.paths.reports / "duplicate_report.jsonl") if (config.paths.reports / "duplicate_report.jsonl").exists() else []
        duplicate_summary = {
            "removed": len(duplicate_records),
            "exact": sum(record.get("duplicate_type") == "exact" for record in duplicate_records),
            "near": sum(record.get("duplicate_type") == "near" for record in duplicate_records),
            "groups": len({record.get("duplicate_group_id") for record in duplicate_records}),
            "impact_percent": round(len(duplicate_records) / max(1, len(documents) + len(duplicate_records)) * 100.0, 6),
        }
        balance_report = {**balance.report, "excluded_documents": [item.to_dict() for item in balance.exclusions]}
        atomic_write_json(temporary / "rejection_summary.json", rejection_summary)
        atomic_write_json(temporary / "duplicate_summary.json", duplicate_summary)
        atomic_write_json(temporary / "balance_report.json", balance_report)

        by_source = Counter(document.source_name for document in balance.selected)
        by_category = Counter(document.category for document in balance.selected)
        by_license = Counter(document.license for document in balance.selected)
        raw_path = config.paths.raw / "documents.jsonl"
        raw_document_count = sum(1 for _ in iter_jsonl(raw_path)) if raw_path.exists() else len(documents)
        representative_samples: dict[str, list[dict[str, Any]]] = {}
        for category in sorted(by_category):
            candidates = sorted(
                (document for document in balance.selected if document.category == category),
                key=lambda document: hashlib.sha256(f"{selected_seed}:{document.document_id}".encode("utf-8")).hexdigest(),
            )
            representative_samples[category] = [
                {
                    "document_id": document.document_id,
                    "source_name": document.source_name,
                    "text_preview": document.text[:500],
                }
                for document in candidates[:2]
            ]
        sources_requiring_review = sorted(
            source.source_name for source in registry.all_sources()
            if source.source_name in selected_by_source and (
                source.license == "REVIEW_REQUIRED"
                or config.license_policy.rule_for(source.license) is None
                or config.license_policy.rule_for(source.license).status != "allowed"  # type: ignore[union-attr]
                or not config.dataset_mode.release_cleared
            )
        )
        dataset_summary = {
            "schema_version": 1,
            "raw_documents": raw_document_count,
            "cleaned_documents_before_balancing": len(documents),
            "documents": len(balance.selected),
            "estimated_pre_tokenizer_tokens": balance.report["estimated_pre_tokenizer_tokens"],
            "exact_candidate_token_counts": {},
            "by_split": {name: len(records) for name, records in split_map.items()},
            "by_source": dict(sorted(by_source.items())),
            "by_category": dict(sorted(by_category.items())),
            "by_license": dict(sorted(by_license.items())),
            "rejection_distribution": rejection_summary,
            "deduplication_impact": duplicate_summary,
            "balancing_exclusions": balance.report["balancing_exclusions_by_reason"],
            "sensitive_data_detections": rejection_summary["sensitive_data_detections"],
            "source_concentration_percentages": balance.report["source_concentration_percentages"],
            "largest_source_concentration_percent": max(balance.report["source_concentration_percentages"].values(), default=0.0),
            "stated_license_and_terms_distribution": dict(sorted(by_license.items())),
            "sources_marked_review_required": sorted(
                source.source_name for source in registry.all_sources()
                if source.source_name in selected_by_source and source.license == "REVIEW_REQUIRED"
            ),
            "sources_requiring_review_before_open_weight_release": sources_requiring_review,
            "representative_accepted_samples": representative_samples,
            "dataset_mode": config.dataset_mode.to_dict(),
            "token_count_note": "Pre-tokenizer values are provisional estimates, not exact tokenizer counts.",
        }
        atomic_write_json(temporary / "dataset_summary.json", dataset_summary)

        budget_configuration_path = config.paths.configuration / (
            "research_budget.json" if config.paths.collection is not None else "pilot_budget.json"
        )
        configuration_paths = (
            config.paths.configuration / "data_pipeline.json",
            budget_configuration_path,
            config.paths.configuration / "tokenizer.json",
            config.paths.configuration / "dataset_mode.json",
        )
        source_review_path = config.paths.configuration / "approved_sources.json"
        local_source_path = config.paths.configuration / "local_research_sources.json"
        license_policy_path = config.paths.configuration / "license_policy.json"
        collection_source_path = config.paths.collection_source_config
        input_paths = [deduplicated_path, source_review_path, local_source_path, license_policy_path, *configuration_paths]
        if collection_source_path is not None and collection_source_path.exists():
            input_paths.append(collection_source_path)
        input_hashes = {str(path.relative_to(config.paths.project_root)): sha256_file(path) for path in input_paths}
        output_names = [
            "source_manifest.jsonl", "license_manifest.jsonl", "train_manifest.jsonl",
            "validation_manifest.jsonl", "test_manifest.jsonl", "rejection_summary.json",
            "duplicate_summary.json", "balance_report.json", "dataset_summary.json",
            "LOCAL_RESEARCH_ONLY.txt",
        ]
        limitations = known_limitations or [
            "This private local-research snapshot is not cleared for public dataset or model-weight release.",
            "Stated licenses and terms labels require source-by-source review before any publication.",
            "Provisional token estimates are not exact production-tokenizer counts.",
        ]
        local_notice = (
            "LOCAL RESEARCH ONLY\n\n"
            "This snapshot is not cleared for dataset redistribution or public model-weight release.\n"
            "Local experimentation does not eliminate copyright, license, attribution, privacy, or other obligations.\n"
            "Review and, where necessary, remove every source before any publication.\n"
        )
        atomic_write_text(temporary / "LOCAL_RESEARCH_ONLY.txt", local_notice)
        output_hashes = {filename: sha256_file(temporary / filename) for filename in output_names}
        stable_identity_payload = {
            "name": name,
            "version": version,
            "seed": selected_seed,
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "pipeline_configuration_hash": _hash_payload(config.pilot_fingerprint_payload()),
        }
        manifest = {
            "schema_version": 1,
            "snapshot_name": name,
            "snapshot_version": version,
            "creation_timestamp": utc_now(),
            "git_commit": _git_commit(config.paths.project_root),
            "pipeline_configuration_hash": _hash_payload(config.pilot_fingerprint_payload()),
            "source_review_configuration_hash": _hash_payload({
                "audited_sources": sha256_file(source_review_path),
                "local_research_sources": sha256_file(local_source_path),
                "collection_source_selection": (
                    sha256_file(collection_source_path)
                    if collection_source_path is not None and collection_source_path.exists() else None
                ),
            }),
            "license_policy_hash": sha256_file(license_policy_path),
            "seed": selected_seed,
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "snapshot_content_hash": _hash_payload(stable_identity_payload),
            "accepted_document_count": len(balance.selected),
            "rejected_document_count": rejection_summary["total"] + len(balance.exclusions),
            "estimated_token_count": balance.report["estimated_pre_tokenizer_tokens"],
            "source_distribution": dict(sorted(by_source.items())),
            "category_distribution": dict(sorted(by_category.items())),
            "license_distribution": dict(sorted(by_license.items())),
            "known_limitations": limitations,
            "production_readiness_status": "pilot_only",
            "fixture_artifact": all(document.source_name == "sample" for document in balance.selected),
            "local_research_only": config.dataset_mode.local_research_only,
            "release_cleared": config.dataset_mode.release_cleared,
            "production_ready": False,
            "weight_publication_allowed": config.dataset_mode.weight_publication_allowed,
            "dataset_redistribution_allowed": config.dataset_mode.dataset_redistribution_allowed,
            "sources_requiring_review_before_open_weight_release": sources_requiring_review,
            "immutable": True,
        }
        atomic_write_json(temporary / "snapshot_manifest.json", manifest)
        checksum_names = [name for name in SNAPSHOT_FILES if name != "checksums.sha256"]
        checksum_text = "".join(f"{sha256_file(temporary / filename)}  {filename}\n" for filename in checksum_names)
        atomic_write_text(temporary / "checksums.sha256", checksum_text)
        os.replace(temporary, target)
        temporary = None
        return {
            "status": "complete",
            "snapshot": name,
            "snapshot_version": version,
            "snapshot_directory": directory_name,
            "path": str(target),
            "documents": len(balance.selected),
            "estimated_pre_tokenizer_tokens": balance.report["estimated_pre_tokenizer_tokens"],
            "snapshot_content_hash": manifest["snapshot_content_hash"],
            "collection_end_reason": balance.report["collection_end_reason"],
            "files": list(SNAPSHOT_FILES),
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def verify_snapshot(snapshot_directory: Path) -> dict[str, Any]:
    manifest_path = snapshot_directory / "snapshot_manifest.json"
    checksum_path = snapshot_directory / "checksums.sha256"
    if not manifest_path.exists() or not checksum_path.exists():
        raise ValueError("snapshot manifest or checksum file is missing")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        path = snapshot_directory / filename
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"frozen snapshot checksum mismatch: {filename}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("immutable"):
        raise ValueError("snapshot is not marked immutable")
    return manifest
