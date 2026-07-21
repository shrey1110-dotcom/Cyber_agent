from __future__ import annotations

from cyber_agent.data_pipeline.deduplicate import deduplicate_documents
from cyber_agent.data_pipeline.schemas import Document, sha256_text
from cyber_agent.data_pipeline.split import split_documents


def document(identifier: str, text: str, group: str | None = None) -> Document:
    metadata = {"duplicate_group_id": group} if group else {}
    return Document(
        document_id=identifier,
        text=text,
        source_name="fixture",
        source_url=f"fixture://{identifier}",
        license="CC0-1.0",
        category="cybersecurity",
        language="en",
        retrieved_at="2026-07-20T00:00:00+00:00",
        content_hash=sha256_text(text),
        quality_score=0.9,
        metadata=metadata,
    )


BASE = "Incident response preserves evidence and records a reliable timeline. Analysts isolate confirmed threats and collect volatile data with approved tools. Every action is logged for independent review."


def test_exact_and_near_duplicate_grouping() -> None:
    near = BASE.replace("independent review", "later independent review")
    unique = "Linux service inspection records process identifiers, executable paths, open sockets, and timestamps before an approved change is made."
    retained, removed = deduplicate_documents(
        [document("a", BASE), document("b", BASE), document("c", near), document("d", unique)],
        hamming_threshold=16,
        minimum_shingle_similarity=0.55,
    )
    assert len(retained) == 2
    assert {record.duplicate_type for record in removed} == {"exact", "near"}
    grouped = next(item for item in retained if item.document_id == "a")
    assert grouped.metadata["duplicate_group_size"] == 3


def test_deterministic_split_and_duplicate_leakage_prevention() -> None:
    documents = [document(f"doc-{index:02d}", f"Unique English security document number {index} with enough distinct information for deterministic assignment.") for index in range(30)]
    documents.extend(
        [
            document("group-a", "First related document with distinct text.", "shared-group"),
            document("group-b", "Second related document with distinct text.", "shared-group"),
        ]
    )
    proportions = {"train": 0.6, "validation": 0.2, "test": 0.2}
    first_splits, first_manifest = split_documents(documents, seed=42, proportions=proportions)
    second_splits, second_manifest = split_documents(list(reversed(documents)), seed=42, proportions=proportions)
    assert [item.to_dict() for item in first_manifest] == [item.to_dict() for item in second_manifest]
    shared_assignments = {item.split for item in first_manifest if item.duplicate_group_id == "shared-group"}
    assert len(shared_assignments) == 1
    hashes_by_split = {
        name: {item.content_hash for item in records}
        for name, records in first_splits.items()
    }
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["validation"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["test"])
    assert hashes_by_split["validation"].isdisjoint(hashes_by_split["test"])

