"""Train-only streaming token blocks from an immutable Phase 3.5 snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx

from cyber_agent.data_pipeline.export import read_jsonl
from cyber_agent.data_pipeline.snapshot import verify_snapshot
from cyber_agent.tokenizer.artifacts import verify_candidate
from cyber_agent.tokenizer.corpus import sha256_file
from cyber_agent.tokenizer.loader import CyberTokenizer


def _inside(root: Path, value: Path) -> Path:
    resolved = value.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("training artifact path must remain inside the project root") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    project_root: Path
    snapshot_name: str
    snapshot_directory: Path
    tokenizer_directory: Path
    tokenizer_path: Path
    snapshot_manifest: dict[str, Any]
    tokenizer_manifest: dict[str, Any]
    tokenizer: CyberTokenizer
    train_manifest_hash: str
    validation_manifest_hash: str
    test_manifest_hash: str
    document_limit: int | None = None

    @classmethod
    def load(
        cls,
        *,
        project_root: Path,
        snapshot_name: str,
        tokenizer_path: str | Path,
        allow_pilot_artifacts: bool,
        document_limit: int | None = None,
    ) -> "TrainingArtifacts":
        root = project_root.resolve()
        snapshot_directory = _inside(root, root / "artifacts" / "datasets" / "snapshots" / snapshot_name)
        snapshot = verify_snapshot(snapshot_directory)
        tokenizer_file = _inside(root, Path(tokenizer_path))
        tokenizer_directory = tokenizer_file.parent
        tokenizer_manifest = verify_candidate(tokenizer_directory)
        tokenizer = CyberTokenizer.from_file(tokenizer_file)
        if tokenizer_manifest.get("snapshot_content_hash") != snapshot.get("snapshot_content_hash"):
            raise ValueError("tokenizer was not trained from the requested immutable snapshot")
        if tokenizer.vocabulary_size != int(tokenizer_manifest.get("actual_vocabulary_size", -1)):
            raise ValueError("tokenizer vocabulary metadata does not match tokenizer.json")
        if bool(snapshot.get("local_research_only")) or not bool(snapshot.get("release_cleared")):
            if not allow_pilot_artifacts:
                raise ValueError(
                    "this snapshot is local-research/pilot-only; pass --allow-pilot-artifacts for a non-production local run"
                )
        if not bool(tokenizer_manifest.get("production_ready")) and not allow_pilot_artifacts:
            raise ValueError("tokenizer is not production-ready; pass --allow-pilot-artifacts for a local smoke/pilot run")
        train_path = snapshot_directory / "train_manifest.jsonl"
        validation_path = snapshot_directory / "validation_manifest.jsonl"
        test_path = snapshot_directory / "test_manifest.jsonl"
        if not all(path.exists() for path in (train_path, validation_path, test_path)):
            raise ValueError("immutable snapshot is missing one or more split manifests")
        return cls(
            project_root=root,
            snapshot_name=snapshot_name,
            snapshot_directory=snapshot_directory,
            tokenizer_directory=tokenizer_directory,
            tokenizer_path=tokenizer_file,
            snapshot_manifest=snapshot,
            tokenizer_manifest=tokenizer_manifest,
            tokenizer=tokenizer,
            train_manifest_hash=sha256_file(train_path),
            validation_manifest_hash=sha256_file(validation_path),
            test_manifest_hash=sha256_file(test_path),
            document_limit=document_limit,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "snapshot_name": self.snapshot_name,
            "snapshot_content_hash": self.snapshot_manifest["snapshot_content_hash"],
            "snapshot_manifest_hash": sha256_file(self.snapshot_directory / "snapshot_manifest.json"),
            "train_manifest_hash": self.train_manifest_hash,
            "validation_manifest_hash": self.validation_manifest_hash,
            "test_manifest_hash": self.test_manifest_hash,
            "tokenizer_path": str(self.tokenizer_path.relative_to(self.project_root)),
            "tokenizer_hash": sha256_file(self.tokenizer_path),
            "tokenizer_training_manifest_hash": sha256_file(self.tokenizer_directory / "training_manifest.json"),
            "tokenizer_vocabulary_size": self.tokenizer.vocabulary_size,
            "train_only_verified": True,
            "local_research_only": bool(self.snapshot_manifest.get("local_research_only")),
            "release_cleared": bool(self.snapshot_manifest.get("release_cleared")),
            "weight_publication_allowed": bool(self.snapshot_manifest.get("weight_publication_allowed")),
        }

    def document_count(self, split: str) -> int:
        return sum(1 for _ in self._records(split))

    def _records(self, split: str) -> Iterator[dict[str, Any]]:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        path = self.snapshot_directory / f"{split}_manifest.jsonl"
        seen: set[str] = set()
        for line_number, record in enumerate(read_jsonl(path), start=1):
            if self.document_limit is not None and len(seen) >= self.document_limit:
                break
            if record.get("split") != split:
                raise ValueError(f"{split} manifest has a non-{split} record at line {line_number}")
            document_id = str(record.get("document_id", ""))
            text = record.get("text")
            if not document_id or document_id in seen or not isinstance(text, str) or not text:
                raise ValueError(f"invalid {split} document at line {line_number}")
            seen.add(document_id)
            yield record

    def token_blocks(self, split: str, *, sequence_length: int) -> Iterator[list[int]]:
        """Stream fixed blocks without ever mixing validation/test into training.

        Literal control-token strings in documents remain ordinary content because
        ``CyberTokenizer.encode`` defaults to ``parse_special_tokens=False``.
        """
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        block_size = sequence_length + 1
        carry: list[int] = []
        for record in self._records(split):
            text = str(record["text"])
            document_tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if not document_tokens:
                continue
            carry.extend(document_tokens)
            while len(carry) >= block_size:
                yield carry[:block_size]
                carry = carry[block_size:]

    def batches(self, split: str, *, sequence_length: int, batch_size: int) -> Iterator[tuple[mx.array, mx.array]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batch: list[list[int]] = []
        for block in self.token_blocks(split, sequence_length=sequence_length):
            batch.append(block)
            if len(batch) == batch_size:
                values = mx.array(batch, dtype=mx.int32)
                yield values[:, :-1], values[:, 1:]
                batch = []

    def data_fingerprint(self) -> str:
        payload = json.dumps(self.provenance(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
