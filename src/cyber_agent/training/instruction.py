"""Auditable local instruction-pilot data for a bounded chat adaptation.

This module is intentionally separate from the frozen corpus used for
pretraining.  It accepts only an explicitly supplied, project-contained JSONL
file with a documented source and license, and it never reads the frozen
validation or test manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx

from cyber_agent.data_pipeline.export import read_jsonl
from cyber_agent.tokenizer.corpus import sha256_file
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.training.data import TrainingArtifacts


ALLOWED_SPLITS = frozenset({"train", "validation"})
REQUIRED_FIELDS = frozenset(
    {"record_id", "split", "category", "system", "user", "assistant", "license", "source_name"}
)


def _inside(root: Path, value: Path) -> Path:
    resolved = value.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("instruction dataset must remain inside the project root") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class InstructionRecord:
    """One project-authored supervised prompt/completion pair.

    The literal control-token spellings are never parsed while rendering this
    record.  Only the tokenizer's explicit BOS/EOS insertion is trusted.
    """

    record_id: str
    split: str
    category: str
    system: str
    user: str
    assistant: str
    license: str
    source_name: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, line_number: int) -> "InstructionRecord":
        missing = sorted(REQUIRED_FIELDS - set(value))
        if missing:
            raise ValueError(f"instruction record {line_number} is missing fields: {', '.join(missing)}")
        record = cls(**{field: str(value[field]) for field in REQUIRED_FIELDS})
        if record.split not in ALLOWED_SPLITS:
            raise ValueError(f"instruction record {line_number} has an invalid split")
        if record.license != "MIT":
            raise ValueError(f"instruction record {line_number} must declare the reviewed MIT license")
        if record.source_name != "cyber-agent-authored-instruction-pilot":
            raise ValueError(f"instruction record {line_number} has an unapproved source_name")
        if not record.record_id or not record.category:
            raise ValueError(f"instruction record {line_number} has an empty identifier or category")
        if not record.system.strip() or not record.user.strip() or not record.assistant.strip():
            raise ValueError(f"instruction record {line_number} contains empty prompt or completion text")
        return record

    def render(self) -> str:
        """Render the same ordinary-text format used by the v0 chat runtime."""
        return f"System: {self.system}\n\nUser: {self.user}\n\nAssistant: {self.assistant}"


@dataclass(frozen=True, slots=True)
class InstructionTrainingArtifacts:
    """Streaming, split-isolated artifacts for local instruction adaptation."""

    project_root: Path
    base: TrainingArtifacts
    dataset_path: Path
    dataset_hash: str
    records: tuple[InstructionRecord, ...]

    @classmethod
    def load(
        cls,
        *,
        project_root: Path,
        base: TrainingArtifacts,
        dataset_path: str | Path,
    ) -> "InstructionTrainingArtifacts":
        root = project_root.resolve()
        path = _inside(root, Path(dataset_path))
        if not path.is_file():
            raise ValueError("instruction dataset is missing or is not a regular file")
        records = tuple(
            InstructionRecord.from_dict(record, line_number=line_number)
            for line_number, record in enumerate(read_jsonl(path), start=1)
        )
        if not records:
            raise ValueError("instruction dataset is empty")
        record_ids = [record.record_id for record in records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("instruction dataset contains duplicate record_id values")
        counts = {split: sum(record.split == split for record in records) for split in ALLOWED_SPLITS}
        if not counts["train"] or not counts["validation"]:
            raise ValueError("instruction dataset requires non-empty train and validation splits")
        return cls(
            project_root=root,
            base=base,
            dataset_path=path,
            dataset_hash=sha256_file(path),
            records=records,
        )

    @property
    def tokenizer(self) -> CyberTokenizer:
        return self.base.tokenizer

    def provenance(self) -> dict[str, Any]:
        # Preserve the original frozen corpus/tokenizer provenance so the chat
        # loader can verify the base model exactly, then add SFT-specific facts.
        return {
            **self.base.provenance(),
            "data_mode": "local_instruction_adaptation",
            "instruction_dataset_path": str(self.dataset_path.relative_to(self.project_root)),
            "instruction_dataset_hash": self.dataset_hash,
            "instruction_source_name": "cyber-agent-authored-instruction-pilot",
            "instruction_license": "MIT",
            "instruction_split_counts": {
                split: sum(record.split == split for record in self.records) for split in sorted(ALLOWED_SPLITS)
            },
            "instruction_train_only_verified": True,
        }

    def _records(self, split: str) -> Iterator[InstructionRecord]:
        if split not in ALLOWED_SPLITS:
            raise ValueError("instruction split must be train or validation")
        yield from (record for record in self.records if record.split == split)

    def token_blocks(self, split: str, *, sequence_length: int) -> Iterator[list[int]]:
        """Render complete examples for inspection without special-token parsing."""
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        maximum = sequence_length + 1
        for record in self._records(split):
            # parse_special_tokens remains false: user-like content cannot
            # create a trusted control boundary during supervised training.
            tokens = self.tokenizer.encode(record.render(), add_bos=True, add_eos=True, parse_special_tokens=False)
            if len(tokens) > maximum:
                raise ValueError(
                    f"instruction record {record.record_id} exceeds the configured context length; do not truncate it"
                )
            yield tokens

    def _supervised_example(self, record: InstructionRecord, *, sequence_length: int) -> tuple[list[int], list[int], list[float]]:
        """Return fixed-size inputs, targets, and an assistant-only loss mask."""
        prompt = f"System: {record.system}\n\nUser: {record.user}\n\nAssistant: "
        prefix = self.tokenizer.encode(prompt, add_bos=True, parse_special_tokens=False)
        completion = self.tokenizer.encode(record.assistant, parse_special_tokens=False)
        sequence = [*prefix, *completion, self.tokenizer.eos_token_id]
        maximum = sequence_length + 1
        if len(sequence) > maximum:
            raise ValueError(
                f"instruction record {record.record_id} exceeds the configured context length; do not truncate it"
            )
        inputs = sequence[:-1]
        targets = sequence[1:]
        # targets[k] is sequence[k + 1], so the first supervised completion
        # target follows the final prompt token at prefix length - 1.
        weights = [0.0] * (len(prefix) - 1) + [1.0] * (len(sequence) - len(prefix))
        if len(weights) != len(targets):
            raise AssertionError("instruction target mask is misaligned")
        padding = sequence_length - len(inputs)
        return (
            [*inputs, *([self.tokenizer.pad_token_id] * padding)],
            [*targets, *([self.tokenizer.pad_token_id] * padding)],
            [*weights, *([0.0] * padding)],
        )

    def batches(
        self, split: str, *, sequence_length: int, batch_size: int
    ) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batch: list[tuple[list[int], list[int], list[float]]] = []
        for record in self._records(split):
            batch.append(self._supervised_example(record, sequence_length=sequence_length))
            if len(batch) == batch_size:
                inputs, targets, weights = zip(*batch, strict=True)
                yield (
                    mx.array(inputs, dtype=mx.int32),
                    mx.array(targets, dtype=mx.int32),
                    mx.array(weights, dtype=mx.float32),
                )
                batch = []

    def data_fingerprint(self) -> str:
        payload = json.dumps(self.provenance(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
