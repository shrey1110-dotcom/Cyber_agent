"""Allowlisted local-manifest ingestion with fail-closed license controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_jsonl,
    fingerprint,
    stage_is_current,
    write_stage_marker,
)
from cyber_agent.data_pipeline.extract import validate_utf8
from cyber_agent.data_pipeline.schemas import RawDocument, RejectionRecord, sha256_text, stable_document_id, utc_now
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
    input_paths = manifests + [
        item
        for manifest in manifests
        if manifest.parent.exists()
        for item in manifest.parent.rglob("*")
        if item.is_file()
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

    accepted: list[RawDocument] = []
    rejected: list[RejectionRecord] = []
    for source, manifest in zip(sources, manifests, strict=True):
        source_accepted, source_rejected = _ingest_manifest(config, source, manifest)
        accepted.extend(source_accepted)
        rejected.extend(source_rejected)

    accepted.sort(key=lambda record: record.document_id)
    rejected.sort(key=lambda record: record.document_id)
    atomic_write_jsonl(raw_path, (record.to_dict() for record in accepted))
    atomic_write_jsonl(rejected_path, (record.to_dict() for record in rejected))
    counts = {"accepted": len(accepted), "rejected": len(rejected), "sources": len(sources)}
    write_stage_marker(config.paths.manifests, "ingest", input_fingerprint, outputs, counts)
    return {"stage": "ingest", "status": "complete", **counts, "outputs": [str(path) for path in outputs]}


def _ingest_manifest(
    config: PipelineConfig,
    source: SourceDefinition,
    manifest_path: Path,
) -> tuple[list[RawDocument], list[RejectionRecord]]:
    if not manifest_path.exists():
        raise ValueError(f"source manifest does not exist: {manifest_path}")
    try:
        lines = manifest_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read source manifest {manifest_path}: {exc}") from exc
    accepted: list[RawDocument] = []
    rejected: list[RejectionRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fallback_id = stable_document_id(source.source_name, f"{manifest_path.name}#line-{line_number}")
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("manifest record must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            rejected.append(_safe_rejection(fallback_id, source.source_name, "malformed_manifest", str(exc), line_number))
            continue
        source_url = entry.get("source_url")
        document_id = (
            stable_document_id(source.source_name, source_url)
            if isinstance(source_url, str) and source_url.strip()
            else fallback_id
        )
        try:
            record = _load_entry(config, source, manifest_path, entry, document_id)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            message = str(exc)
            code = _rejection_code(message)
            rejected.append(_safe_rejection(document_id, source.source_name, code, message, line_number))
            continue
        accepted.append(record)
    return accepted, rejected


def _load_entry(
    config: PipelineConfig,
    source: SourceDefinition,
    manifest_path: Path,
    entry: dict[str, Any],
    document_id: str,
) -> RawDocument:
    source_url = _required_string(entry, "source_url")
    license_identifier = _required_string(entry, "license")
    config.license_policy.require_allowed(license_identifier)
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
        metadata={**metadata, "source_homepage": source.homepage, "allowed_use": source.allowed_use},
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

