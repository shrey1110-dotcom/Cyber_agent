"""Fail-closed, train-split-only corpus validation and streaming."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import read_jsonl
from cyber_agent.data_pipeline.schemas import Document
from cyber_agent.tokenizer.config import TokenizerConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusDocumentReference:
    document_id: str
    content_hash: str
    source_name: str
    category: str
    license: str
    line_number: int
    character_count: int
    byte_count: int
    selection_priority: int

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "source_name": self.source_name,
            "category": self.category,
            "license": self.license,
            "character_count": self.character_count,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    config: TokenizerConfig
    references: tuple[CorpusDocumentReference, ...]
    input_manifest_hashes: dict[str, str]
    excluded_split_counts: dict[str, int]

    @property
    def document_count(self) -> int:
        return len(self.references)

    @property
    def character_count(self) -> int:
        return sum(reference.character_count for reference in self.references)

    @property
    def byte_count(self) -> int:
        return sum(reference.byte_count for reference in self.references)

    @property
    def source_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reference.source_name for reference in self.references).items()))

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reference.category for reference in self.references).items()))

    @property
    def license_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reference.license for reference in self.references).items()))

    def iter_texts(self) -> Iterator[str]:
        selected_ids = {reference.document_id for reference in self.references}
        yielded: set[str] = set()
        with self.config.training_split_path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                document_id = value.get("document_id")
                if document_id in selected_ids:
                    document = Document.from_dict(value)
                    yielded.add(document.document_id)
                    yield document.text
        missing = selected_ids - yielded
        if missing:
            raise ValueError(f"training documents disappeared during streaming: {len(missing)}")

    def manifest_documents(self) -> list[dict[str, Any]]:
        return [reference.to_manifest_dict() for reference in sorted(self.references, key=lambda item: item.document_id)]


def validate_training_corpus(config: TokenizerConfig) -> CorpusPlan:
    phase2 = PipelineConfig.load(config.project_root)
    expected_train = (config.project_root / "data" / "splits" / "train.jsonl").resolve()
    if config.training_split_path.resolve() != expected_train:
        raise ValueError("tokenizer training is restricted to the Phase 2 training split")

    manifests = config.project_root / "data" / "manifests"
    required_paths = {
        "training_split": config.training_split_path,
        "train_manifest": manifests / "train_manifest.jsonl",
        "source_manifest": manifests / "source_manifest.jsonl",
        "validation_manifest": manifests / "validation_manifest.jsonl",
        "test_manifest": manifests / "test_manifest.jsonl",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise ValueError(f"required Phase 2 tokenizer inputs are missing: {', '.join(missing)}")

    source_records = read_jsonl(required_paths["source_manifest"])
    allowed_sources: dict[str, dict[str, Any]] = {}
    for record in source_records:
        name = record.get("source_name")
        license_identifier = record.get("license")
        if not isinstance(name, str) or not name or not isinstance(license_identifier, str) or not license_identifier:
            raise ValueError("source manifest contains missing source or license metadata")
        phase2.license_policy.require_allowed(license_identifier)
        allowed_sources[name] = record
    if not allowed_sources:
        raise ValueError("source manifest contains no approved sources")

    train_assignments = _load_assignments(required_paths["train_manifest"], expected_split="train")
    validation_assignments = _load_assignments(required_paths["validation_manifest"], expected_split="validation")
    test_assignments = _load_assignments(required_paths["test_manifest"], expected_split="test")
    if set(train_assignments) & (set(validation_assignments) | set(test_assignments)):
        raise ValueError("a training document also appears in validation or test manifests")

    references: list[CorpusDocumentReference] = []
    seen_ids: set[str] = set()
    try:
        handle = config.training_split_path.open("r", encoding="utf-8", errors="strict")
    except OSError as exc:
        raise ValueError(f"cannot open training split: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid training JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"training record at line {line_number} is not an object")
            document = Document.from_dict(value)
            if document.document_id in seen_ids:
                raise ValueError(f"duplicate document in training split: {document.document_id}")
            seen_ids.add(document.document_id)
            if not document.text:
                raise ValueError(f"empty training document: {document.document_id}")
            if len(document.text) > config.maximum_document_characters:
                raise ValueError(f"training document exceeds configured length: {document.document_id}")
            assignment = train_assignments.get(document.document_id)
            if assignment is None:
                raise ValueError(f"training document is absent from train manifest: {document.document_id}")
            if assignment.get("content_hash") != document.content_hash:
                raise ValueError(f"training manifest hash mismatch: {document.document_id}")
            source = allowed_sources.get(document.source_name)
            if source is None:
                raise ValueError(f"training document source is absent from source manifest: {document.source_name}")
            if source["license"] != document.license:
                raise ValueError(f"training document license disagrees with source manifest: {document.document_id}")
            phase2.license_policy.require_allowed(document.license)
            priority = int.from_bytes(
                hashlib.sha256(f"{config.seed}:{document.document_id}".encode("utf-8")).digest()[:8],
                "big",
            )
            references.append(
                CorpusDocumentReference(
                    document.document_id,
                    document.content_hash,
                    document.source_name,
                    document.category,
                    document.license,
                    line_number,
                    len(document.text),
                    len(document.text.encode("utf-8")),
                    priority,
                )
            )
    if seen_ids != set(train_assignments):
        raise ValueError("train manifest and training split document sets differ")

    selected = _apply_quotas(references, config)
    if not selected:
        raise ValueError("training corpus is empty after validation and quotas")
    manifest_hashes = {name: sha256_file(path) for name, path in required_paths.items()}
    return CorpusPlan(
        config,
        tuple(sorted(selected, key=lambda reference: reference.line_number)),
        manifest_hashes,
        {"validation": len(validation_assignments), "test": len(test_assignments)},
    )


def _load_assignments(path: Path, *, expected_split: str) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if record.get("split") != expected_split:
            raise ValueError(f"{path.name} contains a non-{expected_split} assignment")
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{path.name} contains an invalid document_id")
        if document_id in assignments:
            raise ValueError(f"{path.name} repeats document {document_id}")
        assignments[document_id] = record
    return assignments


def _apply_quotas(
    references: list[CorpusDocumentReference],
    config: TokenizerConfig,
) -> list[CorpusDocumentReference]:
    by_source: dict[str, list[CorpusDocumentReference]] = defaultdict(list)
    for reference in references:
        by_source[reference.source_name].append(reference)
    after_source_limits: list[CorpusDocumentReference] = []
    for source_name, source_references in sorted(by_source.items()):
        limit = config.source_sampling_limits.get(source_name, config.source_sampling_limits.get("*"))
        if limit is None:
            raise ValueError(f"no source sampling limit configured for {source_name}")
        ordered = sorted(
            source_references,
            key=lambda reference: (
                reference.selection_priority / config.category_sampling_weights[reference.category],
                reference.document_id,
            ),
        )
        after_source_limits.extend(ordered[:limit])

    by_category: dict[str, list[CorpusDocumentReference]] = defaultdict(list)
    for reference in after_source_limits:
        by_category[reference.category].append(reference)
    selected: list[CorpusDocumentReference] = []
    for category, category_references in sorted(by_category.items()):
        limit = config.category_sampling_limits[category]
        selected.extend(sorted(category_references, key=lambda reference: (reference.selection_priority, reference.document_id))[:limit])
    return selected

