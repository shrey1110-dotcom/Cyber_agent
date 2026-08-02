"""Strongly typed configuration for deterministic local MLX pretraining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for a decoder-only run from random initialization.

    The checked-in default is approximately 50M tied parameters with a 24K
    vocabulary.  A smoke run may derive a smaller configuration, but it is
    always written to the run manifest and can never be mistaken for this
    target architecture.
    """

    schema_version: int
    architecture: str
    vocabulary_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_epsilon: float
    tie_word_embeddings: bool
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_steps: int
    max_steps: int
    checkpoint_every_steps: int
    evaluation_every_steps: int
    checkpoint_directory: str
    compile_steps: bool
    dataset_status: str
    project_root: Path

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, project_root: Path) -> "TrainingConfig":
        required = {
            "schema_version", "architecture", "vocabulary_size", "context_length", "hidden_size",
            "num_layers", "num_attention_heads", "intermediate_size", "rms_norm_epsilon",
            "tie_word_embeddings", "seed", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps", "max_steps",
            "checkpoint_every_steps", "evaluation_every_steps", "checkpoint_directory",
            "compile_steps", "dataset_status",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"training configuration is missing fields: {', '.join(missing)}")
        config = cls(
            schema_version=int(value["schema_version"]),
            architecture=str(value["architecture"]),
            vocabulary_size=int(value["vocabulary_size"]),
            context_length=int(value["context_length"]),
            hidden_size=int(value["hidden_size"]),
            num_layers=int(value["num_layers"]),
            num_attention_heads=int(value["num_attention_heads"]),
            intermediate_size=int(value["intermediate_size"]),
            rms_norm_epsilon=float(value["rms_norm_epsilon"]),
            tie_word_embeddings=bool(value["tie_word_embeddings"]),
            seed=int(value["seed"]),
            batch_size=int(value["batch_size"]),
            gradient_accumulation_steps=int(value["gradient_accumulation_steps"]),
            learning_rate=float(value["learning_rate"]),
            weight_decay=float(value["weight_decay"]),
            max_grad_norm=float(value["max_grad_norm"]),
            warmup_steps=int(value["warmup_steps"]),
            max_steps=int(value["max_steps"]),
            checkpoint_every_steps=int(value["checkpoint_every_steps"]),
            evaluation_every_steps=int(value["evaluation_every_steps"]),
            checkpoint_directory=str(value["checkpoint_directory"]),
            compile_steps=bool(value["compile_steps"]),
            dataset_status=str(value["dataset_status"]),
            project_root=project_root.resolve(),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path, *, project_root: Path | None = None) -> "TrainingConfig":
        config_path = Path(path).resolve()
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load training configuration: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("training configuration must be a JSON object")
        return cls.from_dict(value, project_root=(project_root or config_path.parents[1]))

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "cyber-decoder-v1":
            raise ValueError("unsupported training architecture configuration")
        positive = (
            self.vocabulary_size, self.context_length, self.hidden_size, self.num_layers,
            self.num_attention_heads, self.intermediate_size, self.batch_size,
            self.gradient_accumulation_steps, self.max_steps, self.checkpoint_every_steps,
            self.evaluation_every_steps,
        )
        if any(value < 1 for value in positive):
            raise ValueError("training dimensions and step limits must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate_size must be at least hidden_size")
        if self.rms_norm_epsilon <= 0 or self.learning_rate <= 0:
            raise ValueError("rms_norm_epsilon and learning_rate must be positive")
        if self.weight_decay < 0 or self.max_grad_norm <= 0 or self.warmup_steps < 0:
            raise ValueError("optimizer settings are invalid")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if self.dataset_status not in {"local_research_pilot_only", "production_candidate", "production_frozen"}:
            raise ValueError("dataset_status is not recognized")
        output = self.checkpoint_path
        try:
            output.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("checkpoint_directory must remain inside the project root") from exc

    @property
    def checkpoint_path(self) -> Path:
        return (self.project_root / self.checkpoint_directory).resolve()

    def model_dict(self) -> dict[str, int | float | bool]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "rms_norm_epsilon": self.rms_norm_epsilon,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    def resolved_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["project_root"] = str(self.project_root)
        return value

    def configuration_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.resolved_dict()).encode("utf-8")).hexdigest()

    def with_overrides(self, **changes: Any) -> "TrainingConfig":
        config = replace(self, **changes)
        config.validate()
        return config

    def smoke_config(self, *, vocabulary_size: int, max_steps: int = 2) -> "TrainingConfig":
        """Return a deliberately small architecture for a local mechanics check."""
        return self.with_overrides(
            vocabulary_size=vocabulary_size,
            context_length=min(self.context_length, 64),
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=192,
            batch_size=1,
            warmup_steps=0,
            max_steps=max_steps,
            checkpoint_every_steps=max_steps,
            evaluation_every_steps=max_steps,
            compile_steps=False,
        )
