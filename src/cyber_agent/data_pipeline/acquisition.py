"""Fail-closed, budgeted acquisition and safe archive extraction.

No download is possible without an exact approved source record.  The fixture
workflow does not call this module over the network; tests inject tiny local
responses and archives.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import atomic_write_json
from cyber_agent.data_pipeline.schemas import utc_now
from cyber_agent.data_pipeline.sources import SourceRegistry


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    source_name: str
    exact_release_or_version: str
    url: str
    allowed_domains: tuple[str, ...]
    maximum_bytes: int
    expected_sha256: str | None = None
    timeout_seconds: float = 30.0
    retry_limit: int = 3
    user_agent: str = "cyber-agent-pilot-acquisition/0.1 (+local auditable research pipeline)"
    rate_limit_bytes_per_second: int = 4 * 1024 * 1024

    def validate(self) -> None:
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("approved downloads require an exact HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("download URLs must not embed authentication credentials")
        if parsed.hostname.casefold() not in {domain.casefold() for domain in self.allowed_domains}:
            raise ValueError("download URL domain is outside the approved domain allowlist")
        if self.maximum_bytes < 1 or self.retry_limit < 0 or self.timeout_seconds <= 0:
            raise ValueError("download limits, timeout, and retries must be valid")
        if self.expected_sha256 is not None:
            expected = self.expected_sha256.casefold()
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                raise ValueError("expected_sha256 must be a full hexadecimal SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    maximum_files: int = 10000
    maximum_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    maximum_compression_ratio: float = 100.0

    def validate(self) -> None:
        if self.maximum_files < 1 or self.maximum_uncompressed_bytes < 1 or self.maximum_compression_ratio <= 0:
            raise ValueError("archive extraction limits must be positive")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ApprovedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_domains: tuple[str, ...]) -> None:
        self.allowed_domains = {domain.casefold() for domain in allowed_domains}

    def redirect_request(self, req: urllib.request.Request, fp: BinaryIO, code: int, msg: str, headers: Any, newurl: str):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or not parsed.hostname or parsed.hostname.casefold() not in self.allowed_domains:
            raise ValueError("redirect target is outside the approved HTTPS domain allowlist")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(
    spec: DownloadSpec,
    destination: Path,
    *,
    total_budget_remaining: int | None = None,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Download one exact approved object with resume and atomic publication."""
    spec.validate()
    effective_limit = min(spec.maximum_bytes, total_budget_remaining) if total_budget_remaining is not None else spec.maximum_bytes
    if effective_limit < 1:
        raise ValueError("download budget exhausted before transfer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    request_opener = opener or urllib.request.build_opener(_ApprovedRedirectHandler(spec.allowed_domains)).open
    last_error: Exception | None = None
    resumed_from = temporary.stat().st_size if temporary.exists() else 0
    if resumed_from > effective_limit:
        raise ValueError("partial download already exceeds the configured budget")

    for attempt in range(spec.retry_limit + 1):
        current_size = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": spec.user_agent, "Accept-Encoding": "identity"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        request = urllib.request.Request(spec.url, headers=headers, method="GET")
        try:
            started = time.monotonic()
            response = request_opener(request, timeout=spec.timeout_seconds)
            final_url = getattr(response, "geturl", lambda: spec.url)()
            final_host = urllib.parse.urlparse(final_url).hostname
            if not final_host or final_host.casefold() not in {domain.casefold() for domain in spec.allowed_domains}:
                raise ValueError("response URL is outside the approved domain allowlist")
            status = int(getattr(response, "status", 200))
            append = current_size > 0 and status == 206
            if current_size > 0 and status not in {200, 206}:
                raise ValueError(f"server returned unsupported resume status: {status}")
            content_length_header = response.headers.get("Content-Length")
            if content_length_header:
                response_bytes = int(content_length_header)
                predicted = current_size + response_bytes if append else response_bytes
                if predicted > effective_limit:
                    raise ValueError("declared content length exceeds the configured download budget")
            mode = "ab" if append else "wb"
            written = current_size if append else 0
            with temporary.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > effective_limit:
                        raise ValueError("download exceeded the configured byte budget")
                    handle.write(chunk)
                    if spec.rate_limit_bytes_per_second > 0:
                        expected_elapsed = written / spec.rate_limit_bytes_per_second
                        actual_elapsed = time.monotonic() - started
                        if expected_elapsed > actual_elapsed:
                            time.sleep(min(expected_elapsed - actual_elapsed, 0.25))
                handle.flush()
                os.fsync(handle.fileno())
            digest = sha256_path(temporary)
            if spec.expected_sha256 and digest != spec.expected_sha256.casefold():
                raise ValueError("download checksum does not match the published checksum")
            os.replace(temporary, destination)
            return {
                "source_name": spec.source_name,
                "exact_release_or_version": spec.exact_release_or_version,
                "url": spec.url,
                "final_url": final_url,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "resumed_from_bytes": resumed_from,
                "attempts": attempt + 1,
                "completed_at": utc_now(),
            }
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if isinstance(exc, ValueError) or attempt >= spec.retry_limit:
                break
            time.sleep(min(0.25 * (2**attempt), 2.0))
    assert last_error is not None
    raise ValueError(f"download failed after bounded retries: {last_error}") from last_error


def acquire_reviewed_source(config: PipelineConfig, source_name: str) -> dict[str, Any]:
    """Acquire one explicitly named, enabled, reviewed remote source."""
    registry = SourceRegistry.load(config.paths)
    source = registry.require_downloadable(source_name, config.license_policy)
    parsed = urllib.parse.urlparse(source.download_location)
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError("approved download URL must identify a file")
    spec = DownloadSpec(
        source_name=source.source_name,
        exact_release_or_version=source.exact_release_or_version,
        url=source.download_location,
        allowed_domains=source.approved_domains or (parsed.hostname or "",),
        maximum_bytes=config.pilot_budget.maximum_download_bytes,
        expected_sha256=source.published_sha256 or None,
        timeout_seconds=config.pilot_budget.request_timeout_seconds,
        retry_limit=config.pilot_budget.maximum_retries,
    )
    destination = config.paths.downloads / source.source_name / source.exact_release_or_version / filename
    already_downloaded = sum(
        path.stat().st_size
        for path in config.paths.downloads.rglob("*")
        if path.is_file() and path != destination and not path.name.endswith(".json") and not path.name.startswith(".")
    )
    remaining = config.pilot_budget.maximum_download_bytes - already_downloaded
    result = download_file(spec, destination, total_budget_remaining=remaining)
    manifest_path = destination.parent / "download_manifest.json"
    atomic_write_json(manifest_path, {"schema_version": 1, "downloads": [result]})
    return {**result, "manifest": str(manifest_path)}


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        raise ValueError(f"archive member escapes extraction root: {name}")
    return normalized


def safe_extract_archive(archive: Path, destination: Path, limits: ExtractionLimits | None = None) -> dict[str, Any]:
    """Extract ZIP or TAR data without links, traversal, or unbounded expansion."""
    selected_limits = limits or ExtractionLimits()
    selected_limits.validate()
    if destination.exists():
        raise ValueError("archive destination already exists; extraction is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    file_count = 0
    total_bytes = 0
    compressed_bytes = max(1, archive.stat().st_size)
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                members: list[tuple[PurePosixPath, int, Callable[[], BinaryIO]]] = []
                for info in bundle.infolist():
                    path = _safe_member_path(info.filename)
                    if info.is_dir():
                        continue
                    unix_mode = (info.external_attr >> 16) & 0o170000
                    if unix_mode == 0o120000:
                        raise ValueError("archive symbolic links are not allowed")
                    members.append((path, info.file_size, lambda info=info: bundle.open(info, "r")))
                file_count, total_bytes = _extract_members(members, temporary, compressed_bytes, selected_limits)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, mode="r:*") as bundle:
                members = []
                for info in bundle.getmembers():
                    path = _safe_member_path(info.name)
                    if info.isdir():
                        continue
                    if not info.isfile() or info.issym() or info.islnk():
                        raise ValueError("archive links and special files are not allowed")
                    members.append((path, info.size, lambda info=info: bundle.extractfile(info)))
                file_count, total_bytes = _extract_members(members, temporary, compressed_bytes, selected_limits)
        else:
            raise ValueError("unsupported archive type; only ZIP and TAR are accepted")
        os.replace(temporary, destination)
        temporary = None
        return {"files": file_count, "uncompressed_bytes": total_bytes, "destination": str(destination)}
    finally:
        if temporary is not None and temporary.exists() and temporary.is_dir():
            shutil.rmtree(temporary)


def _extract_members(
    members: list[tuple[PurePosixPath, int, Callable[[], BinaryIO | None]]],
    root: Path,
    compressed_bytes: int,
    limits: ExtractionLimits,
) -> tuple[int, int]:
    if len(members) > limits.maximum_files:
        raise ValueError("archive file-count limit exceeded")
    declared_total = sum(size for _, size, _ in members)
    if declared_total > limits.maximum_uncompressed_bytes:
        raise ValueError("archive decompression byte limit exceeded")
    if declared_total / compressed_bytes > limits.maximum_compression_ratio:
        raise ValueError("archive compression ratio exceeds the decompression-bomb limit")
    written_total = 0
    for relative, declared_size, opener in members:
        target = root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = opener()
        if source is None:
            raise ValueError(f"archive member could not be read: {relative}")
        with source, target.open("xb") as output:
            member_written = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                member_written += len(chunk)
                written_total += len(chunk)
                if member_written > declared_size or written_total > limits.maximum_uncompressed_bytes:
                    raise ValueError("archive expanded beyond declared or configured limits")
                output.write(chunk)
        if member_written != declared_size:
            raise ValueError("archive member size does not match its declaration")
    return len(members), written_total
