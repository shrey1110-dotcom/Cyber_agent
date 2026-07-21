"""Exact SHA-256 and near-duplicate SimHash grouping before splitting."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_jsonl,
    fingerprint,
    read_jsonl,
    stage_is_current,
    write_stage_marker,
)
from cyber_agent.data_pipeline.schemas import Document, DuplicateRecord


def token_shingles(text: str, width: int = 3) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    if len(tokens) < width:
        return set(tokens)
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def simhash64(text: str) -> int:
    """Compute a deterministic 64-bit SimHash over unique three-token shingles."""
    shingles = token_shingles(text)
    if not shingles:
        return 0
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def shingle_similarity(left: str, right: str) -> float:
    left_shingles = token_shingles(left)
    right_shingles = token_shingles(right)
    union = left_shingles | right_shingles
    return len(left_shingles & right_shingles) / len(union) if union else 1.0


class _UnionFind:
    def __init__(self, identifiers: list[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        parent = self.parent[identifier]
        if parent != identifier:
            self.parent[identifier] = self.find(parent)
        return self.parent[identifier]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def deduplicate_documents(
    documents: list[Document],
    *,
    hamming_threshold: int = 12,
    minimum_shingle_similarity: float = 0.60,
) -> tuple[list[Document], list[DuplicateRecord]]:
    ordered = sorted(documents, key=lambda document: document.document_id)
    exact_groups: dict[str, list[Document]] = defaultdict(list)
    for document in ordered:
        exact_groups[document.content_hash].append(document)

    exact_keepers: list[Document] = []
    exact_removed_to_keeper: dict[str, str] = {}
    for group in exact_groups.values():
        keeper = min(group, key=lambda document: document.document_id)
        exact_keepers.append(keeper)
        for duplicate in group:
            if duplicate.document_id != keeper.document_id:
                exact_removed_to_keeper[duplicate.document_id] = keeper.document_id

    union = _UnionFind([document.document_id for document in exact_keepers])
    fingerprints = {document.document_id: simhash64(document.text) for document in exact_keepers}
    pair_similarity: dict[tuple[str, str], float] = {}
    for index, left in enumerate(exact_keepers):
        for right in exact_keepers[index + 1 :]:
            distance = hamming_distance(fingerprints[left.document_id], fingerprints[right.document_id])
            if distance > hamming_threshold:
                continue
            lexical_similarity = shingle_similarity(left.text, right.text)
            if lexical_similarity < minimum_shingle_similarity:
                continue
            union.union(left.document_id, right.document_id)
            pair_similarity[tuple(sorted((left.document_id, right.document_id)))] = lexical_similarity

    near_groups: dict[str, list[Document]] = defaultdict(list)
    by_id = {document.document_id: document for document in exact_keepers}
    for document in exact_keepers:
        near_groups[union.find(document.document_id)].append(document)

    retained: list[Document] = []
    duplicate_records: list[DuplicateRecord] = []
    keeper_to_group: dict[str, str] = {}
    exact_keeper_to_final: dict[str, str] = {}
    for group in near_groups.values():
        final_keeper = min(group, key=lambda document: document.document_id)
        original_members = [document.document_id for document in group]
        original_members.extend(
            removed_id
            for removed_id, exact_keeper_id in exact_removed_to_keeper.items()
            if exact_keeper_id in {document.document_id for document in group}
        )
        group_payload = "\n".join(sorted(original_members)).encode("utf-8")
        group_id = f"dup_{hashlib.sha256(group_payload).hexdigest()}"
        keeper_to_group[final_keeper.document_id] = group_id
        for document in group:
            exact_keeper_to_final[document.document_id] = final_keeper.document_id
            if document.document_id == final_keeper.document_id:
                continue
            similarity = pair_similarity.get(
                tuple(sorted((document.document_id, final_keeper.document_id))),
                shingle_similarity(document.text, final_keeper.text),
            )
            duplicate_records.append(
                DuplicateRecord(document.document_id, final_keeper.document_id, "near", round(similarity, 6), group_id)
            )
        retained.append(
            replace(
                final_keeper,
                metadata={
                    **final_keeper.metadata,
                    "duplicate_group_id": group_id,
                    "duplicate_group_size": len(original_members),
                },
            )
        )

    for removed_id, exact_keeper_id in exact_removed_to_keeper.items():
        final_keeper_id = exact_keeper_to_final[exact_keeper_id]
        group_id = keeper_to_group[final_keeper_id]
        duplicate_records.append(DuplicateRecord(removed_id, final_keeper_id, "exact", 1.0, group_id))

    return (
        sorted(retained, key=lambda document: document.document_id),
        sorted(duplicate_records, key=lambda record: record.removed_document_id),
    )


def run_deduplicate(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    input_path = config.paths.cleaned / "documents.jsonl"
    output_path = config.paths.cleaned / "deduplicated.jsonl"
    report_path = config.paths.reports / "duplicate_report.jsonl"
    input_fingerprint = fingerprint(
        [input_path],
        {"near_duplicate_hamming_distance": config.near_duplicate_hamming_distance},
    )
    outputs = [output_path, report_path]
    if not force and stage_is_current(config.paths.manifests, "deduplicate", input_fingerprint, outputs):
        return {"stage": "deduplicate", "status": "skipped", "outputs": [str(path) for path in outputs]}
    documents = [Document.from_dict(value) for value in read_jsonl(input_path)]
    retained, duplicates = deduplicate_documents(
        documents,
        hamming_threshold=config.near_duplicate_hamming_distance,
    )
    atomic_write_jsonl(output_path, (document.to_dict() for document in retained))
    atomic_write_jsonl(report_path, (record.to_dict() for record in duplicates))
    exact = sum(record.duplicate_type == "exact" for record in duplicates)
    near = sum(record.duplicate_type == "near" for record in duplicates)
    counts = {"input": len(documents), "retained": len(retained), "removed": len(duplicates), "exact": exact, "near": near}
    write_stage_marker(config.paths.manifests, "deduplicate", input_fingerprint, outputs, counts)
    return {"stage": "deduplicate", "status": "complete", **counts, "outputs": [str(path) for path in outputs]}
