from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import py7zr

from cyber_agent.data_pipeline.acquisition import (
    DownloadSpec,
    ExtractionLimits,
    download_file,
    safe_extract_7z_member,
    safe_extract_archive,
    sha1_path,
)
from cyber_agent.data_pipeline.pilot_acquisition import _download_destination, acquisition_lock
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

    disabled_local = next(source for source in registry.all_sources() if source.source_name == "git-2.50.0")
    manifest_path = registry.manifest_path(disabled_local)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disabled placeholder"):
        registry.require_ingestible("git-2.50.0", config.license_policy)


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


def test_download_can_verify_a_published_legacy_sha1_without_using_it_as_identity(tmp_path: Path) -> None:
    url = "https://approved.example/pilot.bin"
    body = b"pinned upstream release"
    destination = tmp_path / "pilot.bin"
    spec = DownloadSpec(
        source_name="fixture",
        exact_release_or_version="v1",
        url=url,
        allowed_domains=("approved.example",),
        maximum_bytes=1024,
        expected_sha1=hashlib.sha1(body).hexdigest(),
        retry_limit=0,
        rate_limit_bytes_per_second=0,
    )

    result = download_file(
        spec,
        destination,
        opener=lambda request, timeout: FakeResponse(body, final_url=url),
    )

    assert result["published_sha1"] == hashlib.sha1(body).hexdigest()
    assert sha1_path(destination) == hashlib.sha1(body).hexdigest()


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


def test_safe_archive_skips_links_without_following_them(tmp_path: Path) -> None:
    archive_path = tmp_path / "links.tar"
    payload = b"package main\n"
    with tarfile.open(archive_path, "w") as archive:
        regular = tarfile.TarInfo("source/main.go")
        regular.size = len(payload)
        archive.addfile(regular, io.BytesIO(payload))
        link = tarfile.TarInfo("source/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)

    report = safe_extract_archive(archive_path, tmp_path / "link-output")

    assert report["files"] == 1
    assert report["skipped_unsafe_members"] == 1
    assert (tmp_path / "link-output" / "source" / "main.go").read_bytes() == payload
    assert not (tmp_path / "link-output" / "source" / "escape").exists()
    assert not (tmp_path / "outside").exists()


def test_safe_7z_extractor_allows_one_declared_member_and_enforces_size(tmp_path: Path) -> None:
    archive = tmp_path / "posts.7z"
    payload = b"<posts><row Id='1' Body='hello'/></posts>"
    with py7zr.SevenZipFile(archive, mode="w") as bundle:
        bundle.writestr(payload, "stackoverflow.com-Posts/Posts.xml")
        bundle.writestr(b"ignore", "stackoverflow.com-Posts/Users.xml")

    output = tmp_path / "extracted" / "Posts.xml"
    report = safe_extract_7z_member(
        archive,
        output,
        member_name="stackoverflow.com-Posts/Posts.xml",
        maximum_uncompressed_bytes=1000,
    )

    assert report["uncompressed_bytes"] == len(payload)
    assert output.read_bytes() == payload
    assert not (tmp_path / "extracted" / "Users.xml").exists()
    with pytest.raises(ValueError, match="already exists"):
        safe_extract_7z_member(
            archive,
            output,
            member_name="stackoverflow.com-Posts/Posts.xml",
            maximum_uncompressed_bytes=1000,
        )
    with pytest.raises(ValueError, match="decompressed-byte limit"):
        safe_extract_7z_member(
            archive,
            tmp_path / "too-small.xml",
            member_name="stackoverflow.com-Posts/Posts.xml",
            maximum_uncompressed_bytes=10,
        )


def test_same_release_archive_reuse_requires_exact_url_and_version(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    registry = SourceRegistry.load(config.paths)
    original = registry.source_by_name("fastapi-0.115.12-en-docs")
    reused = registry.source_by_name("fastapi-0.115.12-code-reuse")

    assert _download_destination(config, reused, registry) == _download_destination(config, original, registry)

    changed_release = replace(reused, exact_release_or_version="0.115.13")
    guarded_registry = SourceRegistry(
        [
            source for source in registry.all_sources()
            if source.source_name != changed_release.source_name
        ] + [changed_release],
        config.paths,
    )
    with pytest.raises(ValueError, match="same exact download URL and pinned release"):
        _download_destination(config, changed_release, guarded_registry)


def test_acquisition_lock_fails_closed_for_an_overlapping_writer(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    with acquisition_lock(config):
        with pytest.raises(ValueError, match="already active or needs recovery"):
            with acquisition_lock(config):
                pass
    assert not (config.paths.manifests / ".acquisition.lock").exists()
