from __future__ import annotations

from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.schemas import RawDocument, sha256_text, stable_document_id
from cyber_agent.data_pipeline.sources import SourceRegistry


def test_document_ids_are_stable_and_source_specific() -> None:
    first = stable_document_id("sample", "fixture://one")
    assert first == stable_document_id("sample", "fixture://one")
    assert first != stable_document_id("sample", "fixture://two")
    assert first != stable_document_id("another", "fixture://one")


def test_content_hash_is_stable_sha256() -> None:
    assert sha256_text("normalized text") == "ab2bb5708d5770916c8e50feef37135ecc1cf2573e52136c3af6cebfb1899aa2"


def test_missing_unknown_and_disallowed_licenses_fail_closed(pipeline_project: Path) -> None:
    policy = PipelineConfig.load(pipeline_project).license_policy
    with pytest.raises(ValueError, match="missing license"):
        policy.require_allowed("")
    with pytest.raises(ValueError, match="unknown license"):
        policy.require_allowed("LicenseRef-Imaginary")
    with pytest.raises(ValueError, match="not allowed"):
        policy.require_allowed("LicenseRef-Proprietary")
    with pytest.raises(ValueError, match="review_required"):
        policy.require_allowed("ODC-BY-1.0")


def test_source_allowlist_enforcement(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    registry = SourceRegistry.load(config.paths)
    assert registry.validate(config.license_policy) == []
    assert registry.require_ingestible("sample", config.license_policy).enabled is True
    with pytest.raises(ValueError, match="not present"):
        registry.require_ingestible("unreviewed-internet", config.license_policy)
    with pytest.raises(ValueError, match="disabled placeholder"):
        registry.require_ingestible("python-documentation-placeholder", config.license_policy)


def test_retrieval_timestamp_must_be_timezone_aware_iso8601() -> None:
    with pytest.raises(ValueError, match="include a timezone"):
        RawDocument(
            document_id="doc_test",
            raw_text="text",
            source_name="sample",
            source_url="fixture://timestamp",
            license="CC0-1.0",
            category="general",
            language="en",
            retrieved_at="2026-07-20T00:00:00",
        )
