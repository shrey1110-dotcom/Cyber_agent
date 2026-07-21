"""Atomic JSON/JSONL I/O, stage markers, and final export helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.schemas import canonical_json, utc_now


def atomic_write_text(
    path: Path,
    text: str,
    *,
    before_replace: Callable[[Path], None] | None = None,
) -> None:
    """Replace a UTF-8 file atomically without damaging a prior good output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace(temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(record) + "\n" for record in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    return records


def fingerprint(paths: Iterable[Path], configuration: Any = None) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        digest.update(str(path).encode("utf-8"))
        if path.exists() and path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    if configuration is not None:
        digest.update(canonical_json(configuration).encode("utf-8"))
    return digest.hexdigest()


def marker_path(manifests_directory: Path, stage: str) -> Path:
    return manifests_directory / "stages" / f"{stage}.json"


def stage_is_current(
    manifests_directory: Path,
    stage: str,
    input_fingerprint: str,
    outputs: Iterable[Path],
) -> bool:
    path = marker_path(manifests_directory, stage)
    if not path.exists() or not all(output.exists() for output in outputs):
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("status") == "complete" and marker.get("input_fingerprint") == input_fingerprint


def write_stage_marker(
    manifests_directory: Path,
    stage: str,
    input_fingerprint: str,
    outputs: Iterable[Path],
    counts: dict[str, int],
) -> None:
    atomic_write_json(
        marker_path(manifests_directory, stage),
        {
            "stage": stage,
            "status": "complete",
            "input_fingerprint": input_fingerprint,
            "outputs": [str(path) for path in outputs],
            "counts": counts,
            "completed_at": utc_now(),
        },
    )


def run_export(config: Any, *, force: bool = False) -> dict[str, Any]:
    """Create dataset, source, rejection, and attribution manifests."""
    # Import locally to keep the low-level atomic I/O helpers dependency-light.
    from cyber_agent.data_pipeline.sources import SourceRegistry

    split_paths = [config.paths.splits / f"{name}.jsonl" for name in ("train", "validation", "test")]
    rejection_paths = [config.paths.rejected / "ingest.jsonl", config.paths.rejected / "clean.jsonl"]
    duplicate_path = config.paths.reports / "duplicate_report.jsonl"
    dataset_path = config.paths.cleaned / "dataset.jsonl"
    source_manifest_path = config.paths.manifests / "source_manifest.jsonl"
    rejection_manifest_path = config.paths.manifests / "rejection_manifest.jsonl"
    outputs = [dataset_path, source_manifest_path, rejection_manifest_path, duplicate_path]
    input_fingerprint = fingerprint(
        [*split_paths, *rejection_paths, duplicate_path],
        config.fingerprint_payload(),
    )
    if not force and stage_is_current(config.paths.manifests, "export", input_fingerprint, outputs):
        return {"stage": "export", "status": "skipped", "outputs": [str(path) for path in outputs]}

    documents = [record for path in split_paths for record in read_jsonl(path)]
    documents.sort(key=lambda record: record["document_id"])
    rejections = [record for path in rejection_paths for record in read_jsonl(path)]
    rejections.sort(key=lambda record: (record["document_id"], record["stage"]))
    counts_by_source: dict[str, int] = {}
    for document in documents:
        counts_by_source[document["source_name"]] = counts_by_source.get(document["source_name"], 0) + 1
    registry = SourceRegistry.load(config.paths)
    used_sources = []
    for source in registry.all_sources():
        if source.source_name not in counts_by_source:
            continue
        used_sources.append(
            {
                "source_name": source.source_name,
                "exact_release_or_version": source.exact_release_or_version,
                "homepage": source.homepage,
                "publisher": source.publisher,
                "data_location": source.data_location,
                "license": source.license,
                "license_evidence_url": source.license_evidence_url,
                "allowed_use": source.allowed_use,
                "redistribution_status": source.redistribution_status,
                "attribution_requirements": source.attribution_requirements,
                "category": source.category,
                "review_status": source.review_status,
                "reviewed_by": source.reviewed_by,
                "reviewed_at": source.reviewed_at,
                "retrieved_at": source.retrieved_at,
                "download_location": source.download_location,
                "local_research_source": source.local_research_source,
                "release_cleared": config.dataset_mode.release_cleared,
                "weight_publication_allowed": config.dataset_mode.weight_publication_allowed,
                "dataset_redistribution_allowed": config.dataset_mode.dataset_redistribution_allowed,
                "document_count": counts_by_source[source.source_name],
            }
        )
    atomic_write_jsonl(dataset_path, documents)
    atomic_write_jsonl(source_manifest_path, used_sources)
    atomic_write_jsonl(rejection_manifest_path, rejections)
    counts = {"documents": len(documents), "sources": len(used_sources), "rejections": len(rejections)}
    write_stage_marker(config.paths.manifests, "export", input_fingerprint, outputs, counts)
    return {"stage": "export", "status": "complete", **counts, "outputs": [str(path) for path in outputs]}
