from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.acquisition import DownloadSpec, ExtractionLimits, download_file, safe_extract_archive
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import read_jsonl
from cyber_agent.data_pipeline.ingest import run_ingest
from cyber_agent.data_pipeline.sources import SourceRegistry


class FakeResponse(io.BytesIO):
    def __init__(self, value: bytes, *, final_url: str, status: int = 200) -> None:
        super().__init__(value)
        self.status = status
        self.headers = {"Content-Length": str(len(value))}
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url


def test_review_status_and_exact_release_enforcement(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    registry = SourceRegistry.load(config.paths)
    sample = registry.require_ingestible("sample", config.license_policy)
    assert sample.review_status == "approved_for_pilot"
    assert sample.exact_release_or_version == "fixture-pilot-v1"
    assert all(
        source.license == "REVIEW_REQUIRED"
        for source in registry.all_sources()
        if source.review_status == "pending" and not source.local_research_source
    )
    with pytest.raises(ValueError, match="disabled placeholder|not approved"):
        registry.require_ingestible("fineweb-edu-placeholder", config.license_policy)

    manifest = pipeline_project / "fixtures" / "sample_corpus" / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["source_release"] = "fixture-pilot-v2"
    lines[0] = json.dumps(first)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = run_ingest(config, ["sample"], force=True)
    assert result["accepted"] == 7
    rejections = read_jsonl(config.paths.rejected / "ingest.jsonl")
    assert any("release does not match" in record["reason"] for record in rejections)


def test_download_budget_redirect_resume_and_atomicity(tmp_path: Path) -> None:
    url = "https://approved.example/pilot.bin"
    full = b"approved-pilot-bytes"
    spec = DownloadSpec(
        source_name="fixture",
        exact_release_or_version="v1",
        url=url,
        allowed_domains=("approved.example",),
        maximum_bytes=1024,
        expected_sha256=hashlib.sha256(full).hexdigest(),
        retry_limit=0,
        rate_limit_bytes_per_second=0,
    )
    destination = tmp_path / "pilot.bin"
    partial = destination.with_name(".pilot.bin.part")
    partial.write_bytes(full[:8])

    def resume_opener(request, timeout):
        assert request.headers["Range"] == "bytes=8-"
        return FakeResponse(full[8:], final_url=url, status=206)

    result = download_file(spec, destination, opener=resume_opener)
    assert destination.read_bytes() == full
    assert result["resumed_from_bytes"] == 8
    assert not partial.exists()

    evil = tmp_path / "evil.bin"
    with pytest.raises(ValueError, match="outside"):
        download_file(
            spec,
            evil,
            opener=lambda request, timeout: FakeResponse(full, final_url="https://evil.example/pilot.bin"),
        )
    assert not evil.exists()

    preserved = tmp_path / "preserved.bin"
    preserved.write_bytes(b"previous")
    too_small = DownloadSpec(
        source_name="fixture", exact_release_or_version="v1", url=url,
        allowed_domains=("approved.example",), maximum_bytes=4, retry_limit=0,
        rate_limit_bytes_per_second=0,
    )
    with pytest.raises(ValueError, match="budget"):
        download_file(
            too_small,
            preserved,
            opener=lambda request, timeout: FakeResponse(full, final_url=url),
        )
    assert preserved.read_bytes() == b"previous"


def test_safe_archive_blocks_traversal_and_decompression_limits(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract_archive(bad, tmp_path / "bad-output")
    assert not (tmp_path / "escape.txt").exists()

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.txt", "A" * 10000)
    with pytest.raises(ValueError, match="decompression byte limit"):
        safe_extract_archive(
            bomb,
            tmp_path / "bomb-output",
            ExtractionLimits(maximum_files=2, maximum_uncompressed_bytes=100, maximum_compression_ratio=1000),
        )

    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("docs/readme.txt", "reviewed fixture")
    report = safe_extract_archive(good, tmp_path / "good-output")
    assert report["files"] == 1
    assert (tmp_path / "good-output" / "docs" / "readme.txt").read_text() == "reviewed fixture"
    with pytest.raises(ValueError, match="already exists"):
        safe_extract_archive(good, tmp_path / "good-output")
