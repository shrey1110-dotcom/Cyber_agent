"""Deterministic duplicate-group-aware document splitting."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_jsonl,
    fingerprint,
    iter_jsonl,
    read_jsonl,
    stage_is_current,
    write_stage_marker,
)
from cyber_agent.data_pipeline.schemas import Document, SplitAssignment, SplitName, canonical_json


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
    balanced_path = config.paths.cleaned / "balanced.jsonl"
    input_path = balanced_path if balanced_path.exists() else config.paths.cleaned / "deduplicated.jsonl"
    split_paths = {
        "train": config.paths.splits / "train.jsonl",
        "validation": config.paths.splits / "validation.jsonl",
        "test": config.paths.splits / "test.jsonl",
    }
    per_split_manifest_paths = {
        name: config.paths.manifests / f"{name}_manifest.jsonl"
        for name in ("train", "validation", "test")
    }
    manifest_path = config.paths.manifests / "split_manifest.jsonl"
    outputs = [*split_paths.values(), *per_split_manifest_paths.values(), manifest_path]
    input_fingerprint = fingerprint(
        [input_path],
        {"seed": selected_seed, "proportions": config.split_proportions},
    )
    if not force and stage_is_current(config.paths.manifests, "split", input_fingerprint, outputs):
        return {"stage": "split", "status": "skipped", "seed": selected_seed, "outputs": [str(path) for path in outputs]}
    temporary_dir = Path(tempfile.mkdtemp(prefix=".split.", suffix=".tmp", dir=config.paths.data))
    temporary_paths = {name: temporary_dir / path.name for name, path in split_paths.items()}
    temporary_manifest = temporary_dir / manifest_path.name
    temporary_per_split = {name: temporary_dir / path.name for name, path in per_split_manifest_paths.items()}
    counts = {"train": 0, "validation": 0, "test": 0}; group_splits: dict[str, str] = {}; seen_hashes: dict[str, str] = {}
    try:
        handles = {name: temporary_paths[name].open("x", encoding="utf-8", newline="\n") for name in split_paths}
        manifests = {name: temporary_per_split[name].open("x", encoding="utf-8", newline="\n") for name in split_paths}
        master = temporary_manifest.open("x", encoding="utf-8", newline="\n")
        try:
            train_boundary = config.split_proportions["train"]; validation_boundary = train_boundary + config.split_proportions["validation"]
            for value in iter_jsonl(input_path):
                document = Document.from_dict(value); group_id = str(document.metadata.get("duplicate_group_id", document.document_id))
                previous_hash_split = seen_hashes.get(document.content_hash)
                if previous_hash_split is not None:
                    split_name = previous_hash_split
                elif group_id in group_splits:
                    split_name = group_splits[group_id]
                else:
                    digest = hashlib.sha256(f"{selected_seed}:{group_id}".encode("utf-8")).digest(); fraction = int.from_bytes(digest[:8], "big") / 2**64
                    split_name = "train" if fraction < train_boundary else "validation" if fraction < validation_boundary else "test"; group_splits[group_id] = split_name
                if previous_hash_split is not None and previous_hash_split != split_name: raise ValueError("content hash leaks across splits")
                seen_hashes[document.content_hash] = split_name
                assignment = SplitAssignment(document.document_id, group_id, split_name, document.source_name, document.content_hash)
                handles[split_name].write(canonical_json(document.to_dict()) + "\n"); encoded = canonical_json(assignment.to_dict()) + "\n"
                manifests[split_name].write(encoded); master.write(encoded); counts[split_name] += 1
            for handle in [*handles.values(), *manifests.values(), master]: handle.flush(); os.fsync(handle.fileno())
        finally:
            for handle in [*handles.values(), *manifests.values(), master]: handle.close()
        for name in split_paths: os.replace(temporary_paths[name], split_paths[name]); os.replace(temporary_per_split[name], per_split_manifest_paths[name])
        os.replace(temporary_manifest, manifest_path)
    finally:
        import shutil; shutil.rmtree(temporary_dir, ignore_errors=True)
    write_stage_marker(config.paths.manifests, "split", input_fingerprint, outputs, counts)
    return {"stage": "split", "status": "complete", "seed": selected_seed, **counts, "outputs": [str(path) for path in outputs]}
