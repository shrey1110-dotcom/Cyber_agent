"""Allowlisted local-manifest ingestion with fail-closed license controls."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import fingerprint, stage_is_current, write_stage_marker
from cyber_agent.data_pipeline.extract import validate_utf8
from cyber_agent.data_pipeline.schemas import RawDocument, RejectionRecord, canonical_json, stable_document_id, utc_now
from cyber_agent.data_pipeline.sources import SourceDefinition, SourceRegistry


def run_ingest(
    config: PipelineConfig,
    source_names: list[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    registry = SourceRegistry.load(config.paths)
    validation_errors = registry.validate(config.license_policy)
    if validation_errors:
        raise ValueError("source configuration is invalid: " + "; ".join(validation_errors))
    sources = (
        [registry.require_ingestible(name, config.license_policy) for name in source_names]
        if source_names
        else registry.enabled_sources(config.license_policy)
    )
    if not sources:
        raise ValueError("no enabled sources selected")

    manifests = [registry.manifest_path(source) for source in sources]
    # Materialized source directories are themselves atomically published.  A
    # full recursive content fingerprint here becomes prohibitively expensive
    # for a 150M-token collection (hundreds of thousands of files) and defeats
    # streaming before ingestion starts.  The manifest plus its source report
    # are the immutable source-stage interface; each content file is still
    # strictly path-checked and UTF-8-validated as it is ingested.
    input_paths = [
        item
        for manifest in manifests
        for item in (manifest, manifest.parent / "materialization_report.json")
    ]
    input_fingerprint = fingerprint(
        input_paths,
        {"sources": [source.source_name for source in sources], **config.fingerprint_payload()},
    )
    raw_path = config.paths.raw / "documents.jsonl"
    rejected_path = config.paths.rejected / "ingest.jsonl"
    outputs = [raw_path, rejected_path]
    if not force and stage_is_current(config.paths.manifests, "ingest", input_fingerprint, outputs):
        return {"stage": "ingest", "status": "skipped", "outputs": [str(path) for path in outputs]}

    # This stage can ingest hundreds of thousands of source documents.  Keep
    # outputs temporary and stream records rather than retaining every raw text
    # and rejection in process memory.
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=".ingest.", suffix=".tmp", dir=config.paths.data))
    raw_temporary = temporary / "documents.jsonl"
    rejected_temporary = temporary / "ingest.jsonl"
    accepted_count = 0
    rejected_count = 0
    budget_end_reasons: dict[str, int] = {}
    try:
        assert temporary is not None
        with raw_temporary.open("x", encoding="utf-8", newline="\n") as raw_handle, rejected_temporary.open(
            "x", encoding="utf-8", newline="\n"
        ) as rejected_handle:
            for source, manifest in sorted(zip(sources, manifests, strict=True), key=lambda item: item[0].source_name):
                source_accepted = 0
                for value in _iter_ingest_manifest(config, source, manifest):
                    if isinstance(value, RejectionRecord):
                        rejected_handle.write(canonical_json(value.to_dict()) + "\n")
                        rejected_count += 1
                        continue
                    if source_accepted >= config.pilot_budget.maximum_documents_per_source:
                        reason = "maximum_documents_per_source"
                    elif accepted_count >= config.pilot_budget.maximum_raw_documents:
                        reason = "maximum_raw_documents"
                    else:
                        raw_handle.write(canonical_json(value.to_dict()) + "\n")
                        source_accepted += 1
                        accepted_count += 1
                        continue
                    budget_end_reasons[reason] = budget_end_reasons.get(reason, 0) + 1
                    rejected_handle.write(
                        canonical_json(
                            RejectionRecord(
                                document_id=value.document_id,
                                source_name=value.source_name,
                                stage="ingest",
                                reason=f"pilot collection stopped at {reason}",
                                reason_codes=(reason,),
                                rejected_at=utc_now(),
                                metadata={"pilot_budget_exclusion": True},
                            ).to_dict()
                        )
                        + "\n"
                    )
                    rejected_count += 1
            raw_handle.flush()
            rejected_handle.flush()
            os.fsync(raw_handle.fileno())
            os.fsync(rejected_handle.fileno())
        os.replace(raw_temporary, raw_path)
        os.replace(rejected_temporary, rejected_path)
        shutil.rmtree(temporary)
        temporary = None
        counts = {"accepted": accepted_count, "rejected": rejected_count, "sources": len(sources)}
        write_stage_marker(config.paths.manifests, "ingest", input_fingerprint, outputs, counts)
        return {
            "stage": "ingest",
            "status": "complete",
            **counts,
            "collection_end_reason": max(budget_end_reasons, key=budget_end_reasons.get) if budget_end_reasons else "input_exhausted",
            "budget_exclusions": dict(sorted(budget_end_reasons.items())),
            "outputs": [str(path) for path in outputs],
        }
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _iter_ingest_manifest(
    config: PipelineConfig,
    source: SourceDefinition,
    manifest_path: Path,
) -> Iterator[RawDocument | RejectionRecord]:
    if not manifest_path.exists():
        raise ValueError(f"source manifest does not exist: {manifest_path}")
    try:
        handle = manifest_path.open(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read source manifest {manifest_path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fallback_id = stable_document_id(source.source_name, f"{manifest_path.name}#line-{line_number}")
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("manifest record must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                yield _safe_rejection(fallback_id, source.source_name, "malformed_manifest", str(exc), line_number)
                continue
            source_url = entry.get("source_url")
            document_id = (
                stable_document_id(source.source_name, source_url)
                if isinstance(source_url, str) and source_url.strip()
                else fallback_id
            )
            try:
                yield _load_entry(config, source, manifest_path, entry, document_id)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                message = str(exc)
                code = _rejection_code(message)
                yield _safe_rejection(document_id, source.source_name, code, message, line_number)


def _load_entry(
    config: PipelineConfig,
    source: SourceDefinition,
    manifest_path: Path,
    entry: dict[str, Any],
    document_id: str,
) -> RawDocument:
    source_url = _required_string(entry, "source_url")
    source_release = _required_string(entry, "source_release")
    if source_release != source.exact_release_or_version:
        raise ValueError(
            f"record release does not match approved release: {source_release} != {source.exact_release_or_version}"
        )
    license_identifier = _required_string(entry, "license")
    config.license_policy.require_usable(
        license_identifier,
        local_research_only=config.dataset_mode.local_research_only,
    )
    if source.license != "MULTIPLE-SPDX-REQUIRED" and license_identifier != source.license:
        raise ValueError(f"record license does not match approved source license: {license_identifier}")
    relative_path = Path(_required_string(entry, "path"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("source file path escapes its manifest directory")
    root = manifest_path.parent.resolve()
    content_path = (root / relative_path).resolve(strict=True)
    try:
        content_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("source file resolves outside its manifest directory") from exc
    raw_bytes = content_path.read_bytes()
    if len(raw_bytes) > config.maximum_document_characters * 4:
        raise ValueError("source file exceeds raw size limit")
    raw_text = validate_utf8(raw_bytes)
    category = entry.get("category", source.category)
    language = entry.get("language", "en")
    retrieved_at = _required_string(entry, "retrieved_at")
    media_type = entry.get("media_type", "text/plain")
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    if category == "code":
        for field_name in ("repository", "revision"):
            value = metadata.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"code record is missing per-record {field_name} metadata")
        declared = metadata.get("detected_licenses", [license_identifier])
        if not isinstance(declared, list) or not declared or any(not isinstance(item, str) for item in declared):
            raise ValueError("code record has ambiguous per-record license metadata")
        if set(declared) != {license_identifier}:
            raise ValueError("code record has conflicting per-record licenses")
    return RawDocument(
        document_id=document_id,
        raw_text=raw_text,
        source_name=source.source_name,
        source_url=source_url,
        license=license_identifier,
        category=category,
        language=language,
        retrieved_at=retrieved_at,
        media_type=media_type,
        attribution_requirements=source.attribution_requirements,
        metadata={
            **metadata,
            "source_homepage": source.homepage,
            "source_release": source.exact_release_or_version,
            "publisher": source.publisher,
            "license_evidence_url": source.license_evidence_url,
            "record_path": str(relative_path),
            "allowed_use": source.allowed_use,
            "local_research_only": config.dataset_mode.local_research_only,
            "source_retrieved_at": source.retrieved_at or retrieved_at,
            "release_cleared": config.dataset_mode.release_cleared,
            "weight_publication_allowed": config.dataset_mode.weight_publication_allowed,
        },
    )


def _required_string(entry: dict[str, Any], name: str) -> str:
    value = entry.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {name}")
    return value


def _rejection_code(message: str) -> str:
    lowered = message.casefold()
    if "license" in lowered:
        return "license_rejected"
    if "utf-8" in lowered:
        return "invalid_utf8"
    if "path" in lowered or "outside" in lowered:
        return "invalid_source_path"
    return "invalid_record"


def _safe_rejection(
    document_id: str,
    source_name: str,
    code: str,
    message: str,
    line_number: int,
) -> RejectionRecord:
    safe_reason = message.splitlines()[0][:240]
    return RejectionRecord(
        document_id=document_id,
        source_name=source_name,
        stage="ingest",
        reason=safe_reason,
        reason_codes=(code,),
        rejected_at=utc_now(),
        metadata={"manifest_line": line_number},
    )
