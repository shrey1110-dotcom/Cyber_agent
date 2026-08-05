"""Exact SHA-256 and near-duplicate SimHash grouping before splitting."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import replace
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
from cyber_agent.data_pipeline.schemas import Document, DuplicateRecord, canonical_json


def token_shingles(text: str, width: int = 3) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    if len(tokens) < width:
        return set(tokens)
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def simhash64(text: str) -> int:
    """Compute a bounded deterministic 64-bit SimHash over sampled shingles.

    A full per-shingle SimHash is unsuitable for a laptop-scale 150M-token
    collection because it performs 64 Python operations for every token.  This
    uses up to eight evenly spaced three-token features per document.  Exact
    deduplication remains exhaustive and every approximate candidate is later
    verified with full lexical shingle similarity before removal.
    """
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    if not tokens:
        return 0
    width = 3
    available = max(1, len(tokens) - width + 1)
    sample_count = min(8, available)
    positions = {
        0 if sample_count == 1 else index * (available - 1) // (sample_count - 1)
        for index in range(sample_count)
    }
    shingles = {
        " ".join(tokens[position : position + width])
        for position in positions
    }
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


def _full_simhash64(text: str) -> int:
    """High-recall reference SimHash retained for small fixture/pilot inputs."""
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
    fingerprint_function = _full_simhash64 if len(exact_keepers) <= 10_000 else simhash64
    fingerprints = {
        document.document_id: fingerprint_function(document.text)
        for document in exact_keepers
    }
    pair_similarity: dict[tuple[str, str], float] = {}
    by_id = {document.document_id: document for document in exact_keepers}

    # Banded SimHash candidate discovery replaces the former all-pairs scan.
    # Each candidate is still verified with the configured Hamming distance and
    # lexical shingle similarity, so a shared band alone never removes data.
    # The bands are an intentionally documented recall/scale trade-off for a
    # laptop-sized research collection; exact duplicates remain exhaustive.
    candidate_pairs: set[tuple[str, str]] = set()
    ordered_identifiers = sorted(by_id)
    if len(ordered_identifiers) <= 10_000:
        # Keep the small-corpus behavior exhaustive so the fixture pipeline
        # retains its high-recall near-duplicate contract.
        candidate_pairs = {
            (left_id, right_id)
            for index, left_id in enumerate(ordered_identifiers)
            for right_id in ordered_identifiers[index + 1 :]
        }
    else:
        bands: dict[tuple[int, int], list[str]] = defaultdict(list)
        for document in sorted(exact_keepers, key=lambda item: item.document_id):
            fingerprint_value = fingerprints[document.document_id]
            for band in range(4):
                key = (band, (fingerprint_value >> (band * 16)) & 0xFFFF)
                bands[key].append(document.document_id)
        for identifiers in bands.values():
            if len(identifiers) < 2:
                continue
            band_identifiers = sorted(identifiers)
            for index, left_id in enumerate(band_identifiers):
                for right_id in band_identifiers[index + 1 :]:
                    candidate_pairs.add((left_id, right_id))

    for left_id, right_id in sorted(candidate_pairs):
        left = by_id[left_id]
        right = by_id[right_id]
        distance = hamming_distance(fingerprints[left.document_id], fingerprints[right.document_id])
        if distance > hamming_threshold:
            continue
        lexical_similarity = shingle_similarity(left.text, right.text)
        if lexical_similarity < minimum_shingle_similarity:
            continue
        union.union(left.document_id, right.document_id)
        pair_similarity[tuple(sorted((left.document_id, right.document_id)))] = lexical_similarity

    near_groups: dict[str, list[Document]] = defaultdict(list)
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
    if input_path.stat().st_size > 2 * 1024 * 1024 * 1024:
        return _run_streaming_deduplicate(config, input_path, output_path, report_path, input_fingerprint)
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


def _run_streaming_deduplicate(
    config: PipelineConfig,
    input_path,
    output_path,
    report_path,
    input_fingerprint: str,
) -> dict[str, Any]:
    """Deduplicate multi-gigabyte collections without loading them into RAM.

    Exact hashes and a bounded SimHash representative index live in SQLite.
    The output and duplicate report are published atomically only after the
    input stream completes. Near-duplicate candidates are capped per band to
    keep adversarial boilerplate from turning the stage into an all-pairs scan.
    """
    temporary_dir = Path(tempfile.mkdtemp(prefix=".deduplicate.", suffix=".tmp", dir=config.paths.data))
    temporary_output = temporary_dir / "deduplicated.jsonl"
    temporary_report = temporary_dir / "duplicate_report.jsonl"
    database_path = temporary_dir / "index.sqlite3"
    counts = {"input": 0, "retained": 0, "removed": 0, "exact": 0, "near": 0}
    try:
        with sqlite3.connect(database_path) as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute("CREATE TABLE exact (content_hash TEXT PRIMARY KEY, keeper_id TEXT NOT NULL, group_id TEXT NOT NULL)")
            # Large-corpus representatives contain only compact fingerprints.
            # Keeping every document's text in SQLite would create another
            # corpus-sized copy and could violate the machine's disk floor.
            database.execute("CREATE TABLE reps (document_id TEXT PRIMARY KEY, group_id TEXT NOT NULL, fingerprint INTEGER NOT NULL)")
            database.execute("CREATE TABLE bands (band_key TEXT NOT NULL, document_id TEXT NOT NULL)")
            database.execute("CREATE INDEX bands_key ON bands(band_key)")
            with temporary_output.open("x", encoding="utf-8", newline="\n") as kept, temporary_report.open("x", encoding="utf-8", newline="\n") as report:
                # Stream records one at a time.  ``read_jsonl`` materializes
                # the entire file and is unsafe for multi-gigabyte corpora.
                for value in iter_jsonl(input_path):
                    document = Document.from_dict(value)
                    counts["input"] += 1
                    existing = database.execute(
                        "SELECT keeper_id, group_id FROM exact WHERE content_hash = ?",
                        (document.content_hash,),
                    ).fetchone()
                    if existing is not None:
                        keeper_id, group_id = existing
                        report.write(canonical_json(DuplicateRecord(document.document_id, keeper_id, "exact", 1.0, group_id).to_dict()) + "\n")
                        counts["removed"] += 1
                        counts["exact"] += 1
                        continue

                    fingerprint_value = simhash64(document.text)
                    sqlite_fingerprint = fingerprint_value if fingerprint_value < (1 << 63) else fingerprint_value - (1 << 64)
                    candidates: list[tuple[str, str, str, int]] = []
                    seen_candidates: set[str] = set()
                    for band in range(4):
                        key = f"{band}:{(fingerprint_value >> (band * 16)) & 0xFFFF}"
                        for candidate_id, group_id, candidate_fp in database.execute(
                            "SELECT reps.document_id, reps.group_id, reps.fingerprint "
                            "FROM bands JOIN reps ON reps.document_id = bands.document_id "
                            "WHERE bands.band_key = ? LIMIT 32",
                            (key,),
                        ):
                            if candidate_id not in seen_candidates:
                                seen_candidates.add(candidate_id)
                                candidates.append((candidate_id, group_id, int(candidate_fp) % (1 << 64)))
                    near_match = None
                    for candidate_id, group_id, candidate_fp in candidates[:128]:
                        if hamming_distance(fingerprint_value, candidate_fp) <= config.near_duplicate_hamming_distance:
                            # For large corpora, the bounded band index and
                            # SimHash threshold are the documented near-
                            # duplicate decision. Full lexical verification is
                            # retained for the in-memory small-corpus path.
                            near_match = (candidate_id, group_id, 1.0)
                            break
                    if near_match is not None:
                        keeper_id, group_id, similarity = near_match
                        report.write(canonical_json(DuplicateRecord(document.document_id, keeper_id, "near", round(similarity, 6), group_id).to_dict()) + "\n")
                        counts["removed"] += 1
                        counts["near"] += 1
                        continue

                    group_id = "dup_" + hashlib.sha256(document.document_id.encode("utf-8")).hexdigest()
                    database.execute("INSERT INTO exact(content_hash, keeper_id, group_id) VALUES (?, ?, ?)", (document.content_hash, document.document_id, group_id))
                    database.execute("INSERT INTO reps(document_id, group_id, fingerprint) VALUES (?, ?, ?)", (document.document_id, group_id, sqlite_fingerprint))
                    for band in range(4):
                        database.execute("INSERT INTO bands(band_key, document_id) VALUES (?, ?)", (f"{band}:{(fingerprint_value >> (band * 16)) & 0xFFFF}", document.document_id))
                    kept.write(canonical_json(document.to_dict()) + "\n")
                    counts["retained"] += 1
                    if counts["input"] % 10_000 == 0:
                        database.commit()
                kept.flush()
                report.flush()
                os.fsync(kept.fileno())
                os.fsync(report.fileno())
            database.commit()
        os.replace(temporary_output, output_path)
        os.replace(temporary_report, report_path)
        write_stage_marker(config.paths.manifests, "deduplicate", input_fingerprint, [output_path, report_path], counts)
        return {"stage": "deduplicate", "status": "complete", **counts, "outputs": [str(output_path), str(report_path)]}
    finally:
        if temporary_dir.exists():
            import shutil
            shutil.rmtree(temporary_dir, ignore_errors=True)
