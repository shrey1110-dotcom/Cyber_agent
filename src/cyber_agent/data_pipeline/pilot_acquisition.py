"""Orchestrate explicitly configured, capped local-research pilot acquisition."""

from __future__ import annotations

import json
import os
import socket
import urllib.parse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, ParamSpec, TypeVar

from cyber_agent.data_pipeline.acquisition import (
    DownloadSpec,
    ExtractionLimits,
    download_file,
    safe_extract_archive,
)
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.balance import estimate_pre_tokenizer_tokens
from cyber_agent.data_pipeline.export import atomic_write_json, read_jsonl
from cyber_agent.data_pipeline.materialize import (
    materialize_archive,
    materialize_cwe,
    materialize_stix,
    materialize_wikimedia_xml_bz2,
)
from cyber_agent.data_pipeline.sources import SourceDefinition, SourceRegistry
from cyber_agent.data_pipeline.synthetic import EXAMPLE_KINDS, SAFE_TOOLS, generate_safe_tool_examples


P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def acquisition_lock(config: PipelineConfig) -> Iterator[None]:
    """Prevent concurrent acquisition writers for one collection.

    The lock is deliberately not treated as stale automatically.  A stale
    lock is evidence of an interrupted acquisition and must be inspected or
    explicitly recovered, rather than risking two writers publishing the same
    source directory.
    """
    lock = config.paths.manifests / ".acquisition.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        try:
            owner = lock.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "unreadable lock metadata"
        raise ValueError(f"acquisition is already active or needs recovery: {owner}") from exc
    try:
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "collection": config.paths.collection or "default",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def single_acquisition_writer(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(config: PipelineConfig, *args: P.args, **kwargs: P.kwargs) -> R:
        with acquisition_lock(config):
            return function(config, *args, **kwargs)
    return wrapped


def _download_destination(config: PipelineConfig, source: SourceDefinition, registry: SourceRegistry) -> Path:
    reuse_source_name = (source.adapter_options or {}).get("reuse_download_from")
    if reuse_source_name is not None:
        if not isinstance(reuse_source_name, str) or not reuse_source_name:
            raise ValueError(f"source {source.source_name} has an invalid reuse_download_from value")
        original = registry.source_by_name(reuse_source_name)
        if (
            original.download_location != source.download_location
            or original.exact_release_or_version != source.exact_release_or_version
        ):
            raise ValueError("archive reuse requires the same exact download URL and pinned release")
        return _download_destination(config, original, registry)
    filename = Path(urllib.parse.urlparse(source.download_location).path).name
    if not filename:
        raise ValueError(f"configured source URL does not identify a file: {source.source_name}")
    return config.paths.downloads / source.source_name / source.exact_release_or_version / filename


def _materialize_download(
    config: PipelineConfig,
    source: SourceDefinition,
    downloaded: Path,
    *,
    token_limit: int,
) -> dict[str, Any]:
    if source.adapter == "http_stix_json":
        return materialize_stix(config, source, downloaded, token_limit=token_limit)
    if source.adapter == "http_wikimedia_xml_bz2":
        return materialize_wikimedia_xml_bz2(config, source, downloaded, token_limit=token_limit)
    extraction = downloaded.parent / "extracted"
    if not extraction.exists():
        safe_extract_archive(
            downloaded,
            extraction,
            ExtractionLimits(
                maximum_files=config.pilot_budget.maximum_archive_files,
                maximum_uncompressed_bytes=config.pilot_budget.maximum_decompressed_bytes,
                maximum_compression_ratio=100.0,
            ),
        )
    if source.adapter == "http_archive_text":
        return materialize_archive(config, source, extraction, token_limit=token_limit)
    if source.adapter == "http_cwe_xml":
        xml_files = sorted(extraction.rglob("*.xml"))
        if len(xml_files) != 1:
            raise ValueError(f"expected exactly one CWE XML file, found {len(xml_files)}")
        return materialize_cwe(config, source, xml_files[0], token_limit=token_limit)
    raise ValueError(f"unsupported pilot materialization adapter: {source.adapter}")


@single_acquisition_writer
def acquire_pilot(
    config: PipelineConfig,
    *,
    mode: str,
    target_tokens: int,
    seed: int,
    confirm_download: bool,
    source_names: list[str] | None = None,
) -> dict[str, Any]:
    if mode != "local_research_only" or config.dataset_mode.dataset_mode != mode:
        raise ValueError("acquire-pilot requires configured local_research_only mode")
    if target_tokens < 1 or target_tokens > config.pilot_budget.maximum_estimated_tokens:
        raise ValueError("target tokens must be positive and within maximum_estimated_tokens")
    registry = SourceRegistry.load(config.paths)
    errors = registry.validate(config.license_policy)
    if errors:
        raise ValueError("source configuration is invalid: " + "; ".join(errors))
    remote = registry.acquisition_sources(config.license_policy)
    synthetic = registry.synthetic_sources()
    if source_names:
        requested = set(source_names)
        known = {source.source_name for source in (*remote, *synthetic)}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"pilot source is not explicitly configured: {', '.join(unknown)}")
        remote = [source for source in remote if source.source_name in requested]
        synthetic = [source for source in synthetic if source.source_name in requested]
    if remote and not confirm_download:
        raise ValueError("network pilot acquisition requires explicit --confirm-download")

    results: list[dict[str, Any]] = []
    estimated_tokens = 0
    documents = 0
    downloaded_bytes = sum(
        path.stat().st_size for path in config.paths.downloads.rglob("*")
        if path.is_file()
        and "extracted" not in path.parts
        and not path.name.startswith(".")
        and path.name != "download_manifest.json"
    )
    collection_end_reason = "input_exhausted"
    for source in synthetic:
        result = generate_safe_tool_examples(
            source,
            registry.manifest_path(source),
            seed=seed,
            document_count=min(len(SAFE_TOOLS) * len(EXAMPLE_KINDS), config.pilot_budget.maximum_documents_per_source),
        )
        synthetic_rows = read_jsonl(registry.manifest_path(source))
        synthetic_tokens = sum(
            estimate_pre_tokenizer_tokens(
                (registry.manifest_path(source).parent / str(row["path"])).read_text(encoding="utf-8")
            )
            for row in synthetic_rows
        )
        documents += len(synthetic_rows)
        estimated_tokens += synthetic_tokens
        result = {**result, "estimated_pre_tokenizer_tokens": synthetic_tokens}
        results.append({"source_name": source.source_name, "kind": "synthetic", **result})

    for source in remote:
        if estimated_tokens >= target_tokens:
            collection_end_reason = "target_estimated_tokens"
            break
        remaining_download = config.pilot_budget.maximum_download_bytes - downloaded_bytes
        if remaining_download <= 0:
            collection_end_reason = "maximum_download_bytes"
            break
        destination = _download_destination(config, source, registry)
        parsed = urllib.parse.urlparse(source.download_location)
        spec = DownloadSpec(
            source_name=source.source_name,
            exact_release_or_version=source.exact_release_or_version,
            url=source.download_location,
            allowed_domains=source.approved_domains or (parsed.hostname or "",),
            maximum_bytes=min(remaining_download, source.maximum_download_bytes or remaining_download),
            expected_sha256=source.published_sha256 or None,
            timeout_seconds=config.pilot_budget.request_timeout_seconds,
            retry_limit=config.pilot_budget.maximum_retries,
        )
        if destination.exists():
            download_result = {
                "source_name": source.source_name, "status": "skipped",
                "path": str(destination), "bytes": destination.stat().st_size,
                "reason": "already_downloaded",
            }
        else:
            download_result = download_file(spec, destination, total_budget_remaining=remaining_download)
        downloaded_bytes += 0 if download_result.get("status") == "skipped" else int(download_result["bytes"])
        materialized = _materialize_download(
            config,
            source,
            destination,
            token_limit=max(1, target_tokens - estimated_tokens),
        )
        source_tokens = int(materialized.get("estimated_pre_tokenizer_tokens", 0))
        source_documents = int(materialized.get("documents", 0))
        estimated_tokens += source_tokens
        documents += source_documents
        results.append({
            "source_name": source.source_name,
            "kind": "remote",
            "download": download_result,
            "materialization": materialized,
        })
        if documents >= config.pilot_budget.maximum_raw_documents:
            collection_end_reason = "maximum_raw_documents"
            break
        if estimated_tokens >= target_tokens:
            collection_end_reason = "target_estimated_tokens"
            break
        if materialized.get("collection_end_reason") != "input_exhausted":
            collection_end_reason = str(materialized["collection_end_reason"])

    summary = {
        "schema_version": 1,
        "dataset_mode": config.dataset_mode.to_dict(),
        "target_estimated_tokens": target_tokens,
        "seed": seed,
        "explicit_download_confirmation": confirm_download,
        "downloaded_bytes": downloaded_bytes,
        "materialized_documents": documents,
        "materialized_estimated_tokens": estimated_tokens,
        "collection_end_reason": collection_end_reason,
        "results": results,
        "network_sources": [source.source_name for source in remote],
        "hosted_llm_used": False,
        "downloaded_code_executed": False,
    }
    output = config.paths.manifests / "pilot_acquisition.json"
    atomic_write_json(output, summary)
    return {**summary, "manifest": str(output)}
