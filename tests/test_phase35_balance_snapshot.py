from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.balance import balance_documents, estimate_pre_tokenizer_tokens
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.schemas import Document, sha256_text
from cyber_agent.data_pipeline.snapshot import SNAPSHOT_FILES, freeze_snapshot, verify_snapshot
from cyber_agent.provenance import generate_provenance_attestation


def make_document(identifier: str, *, source: str, category: str, text: str, group: str | None = None) -> Document:
    return Document(
        document_id=identifier,
        text=text,
        source_name=source,
        source_url=f"fixture://{identifier}",
        license="CC0-1.0",
        category=category,  # type: ignore[arg-type]
        language="en",
        retrieved_at="2026-07-20T00:00:00+00:00",
        content_hash=sha256_text(text),
        quality_score=1.0,
        metadata={"duplicate_group_id": group or identifier},
    )


def test_provisional_estimator_and_deterministic_balancing(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    budget = replace(
        config.pilot_budget,
        maximum_clean_documents=3,
        maximum_documents_per_source=2,
        maximum_estimated_tokens=10000,
        maximum_tokens_per_source=10000,
        maximum_tokens_per_category={name: 10000 for name in config.pilot_budget.maximum_tokens_per_category},
        minimum_tokens_per_category={name: 0 for name in config.pilot_budget.minimum_tokens_per_category},
    )
    documents = [
        make_document("doc-a", source="one", category="general", text="A technical document " * 20),
        make_document("doc-b", source="one", category="general", text="Another technical document " * 20),
        make_document("doc-c", source="one", category="general", text="Third technical document " * 20),
        make_document("doc-d", source="two", category="code", text="def check():\n    return True\n" * 10),
    ]
    first = balance_documents(documents, budget=budget, seed=42)
    second = balance_documents(list(reversed(documents)), budget=budget, seed=42)
    assert [document.document_id for document in first.selected] == [document.document_id for document in second.selected]
    assert first.report == second.report
    assert len([document for document in first.selected if document.source_name == "one"]) <= 2
    assert len(first.selected) <= 3
    assert first.report["provisional_estimator"]["exact"] is False
    assert estimate_pre_tokenizer_tokens("abcd") == 1
    assert len(first.exclusions) == len(documents) - len(first.selected)


def test_snapshot_is_complete_immutable_and_checksum_verified(tokenizer_project: Path) -> None:
    config = PipelineConfig.load(tokenizer_project)
    result = freeze_snapshot(config, name="fixture-pilot-v1", seed=42)
    snapshot = Path(result["path"])
    assert set(SNAPSHOT_FILES) == {path.name for path in snapshot.iterdir()}
    manifest = verify_snapshot(snapshot)
    assert manifest["snapshot_name"] == "fixture-pilot-v1"
    assert manifest["production_readiness_status"] == "pilot_only"
    assert manifest["local_research_only"] is True
    assert manifest["release_cleared"] is False
    assert manifest["production_ready"] is False
    assert manifest["weight_publication_allowed"] is False
    assert (snapshot / "LOCAL_RESEARCH_ONLY.txt").exists()
    assert manifest["estimated_token_count"] == result["estimated_pre_tokenizer_tokens"]
    assert json.loads((snapshot / "dataset_summary.json").read_text())["exact_candidate_token_counts"] == {}
    with pytest.raises(ValueError, match="cannot be overwritten"):
        freeze_snapshot(config, name="fixture-pilot-v1", seed=42)

    version_two = freeze_snapshot(config, name="fixture-pilot-v1", version=2, seed=42)
    assert Path(version_two["path"]).name == "fixture-pilot-v1.v2"
    assert verify_snapshot(Path(version_two["path"]))["snapshot_version"] == 2
    with pytest.raises(ValueError, match="cannot be overwritten"):
        freeze_snapshot(config, name="fixture-pilot-v1", version=2, seed=42)

    balance_hash = manifest["output_hashes"]["balance_report.json"]
    assert len(balance_hash) == 64
    attestation = generate_provenance_attestation(snapshot, artifact_type="frozen_dataset_snapshot")
    assert attestation.parent == snapshot.parent
    assert json.loads(attestation.read_text())["predicate"]["signature_status"] == "unsigned"
    assert verify_snapshot(snapshot)["snapshot_content_hash"] == manifest["snapshot_content_hash"]
    (snapshot / "balance_report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_snapshot(snapshot)
