"""Deterministic duplicate-group-aware document splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_jsonl,
    fingerprint,
    read_jsonl,
    stage_is_current,
    write_stage_marker,
)
from cyber_agent.data_pipeline.schemas import Document, SplitAssignment, SplitName


def split_documents(
    documents: list[Document],
    *,
    seed: int,
    proportions: dict[str, float],
) -> tuple[dict[SplitName, list[Document]], list[SplitAssignment]]:
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1.0")
    groups: dict[str, list[Document]] = defaultdict(list)
    seen_ids: set[str] = set()
    for document in documents:
        if document.document_id in seen_ids:
            raise ValueError(f"document appears more than once: {document.document_id}")
        seen_ids.add(document.document_id)
        group_id = str(document.metadata.get("duplicate_group_id", document.document_id))
        groups[group_id].append(document)

    split_map: dict[SplitName, list[Document]] = {"train": [], "validation": [], "test": []}
    assignments: list[SplitAssignment] = []
    train_boundary = proportions["train"]
    validation_boundary = train_boundary + proportions["validation"]
    for group_id in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        split_name: SplitName
        if fraction < train_boundary:
            split_name = "train"
        elif fraction < validation_boundary:
            split_name = "validation"
        else:
            split_name = "test"
        for document in sorted(groups[group_id], key=lambda item: item.document_id):
            split_map[split_name].append(document)
            assignments.append(
                SplitAssignment(
                    document.document_id,
                    group_id,
                    split_name,
                    document.source_name,
                    document.content_hash,
                )
            )

    _validate_no_leakage(split_map, assignments)
    return split_map, sorted(assignments, key=lambda assignment: assignment.document_id)


def _validate_no_leakage(
    split_map: dict[SplitName, list[Document]],
    assignments: list[SplitAssignment],
) -> None:
    seen_hashes: dict[str, SplitName] = {}
    for split_name, documents in split_map.items():
        for document in documents:
            previous = seen_hashes.setdefault(document.content_hash, split_name)
            if previous != split_name:
                raise ValueError(f"content hash leaks across {previous} and {split_name}")
    group_splits: dict[str, SplitName] = {}
    for assignment in assignments:
        previous = group_splits.setdefault(assignment.duplicate_group_id, assignment.split)
        if previous != assignment.split:
            raise ValueError("duplicate group leaks across dataset splits")


def run_split(config: PipelineConfig, *, seed: int | None = None, force: bool = False) -> dict[str, Any]:
    selected_seed = config.default_seed if seed is None else seed
    input_path = config.paths.cleaned / "deduplicated.jsonl"
    split_paths = {
        "train": config.paths.splits / "train.jsonl",
        "validation": config.paths.splits / "validation.jsonl",
        "test": config.paths.splits / "test.jsonl",
    }
    manifest_path = config.paths.manifests / "split_manifest.jsonl"
    outputs = [*split_paths.values(), manifest_path]
    input_fingerprint = fingerprint(
        [input_path],
        {"seed": selected_seed, "proportions": config.split_proportions},
    )
    if not force and stage_is_current(config.paths.manifests, "split", input_fingerprint, outputs):
        return {"stage": "split", "status": "skipped", "seed": selected_seed, "outputs": [str(path) for path in outputs]}
    documents = [Document.from_dict(value) for value in read_jsonl(input_path)]
    split_map, assignments = split_documents(
        documents,
        seed=selected_seed,
        proportions=config.split_proportions,
    )
    for name, path in split_paths.items():
        atomic_write_jsonl(path, (document.to_dict() for document in split_map[name]))
    atomic_write_jsonl(manifest_path, (assignment.to_dict() for assignment in assignments))
    counts = {name: len(documents_in_split) for name, documents_in_split in split_map.items()}
    write_stage_marker(config.paths.manifests, "split", input_fingerprint, outputs, counts)
    return {"stage": "split", "status": "complete", "seed": selected_seed, **counts, "outputs": [str(path) for path in outputs]}

