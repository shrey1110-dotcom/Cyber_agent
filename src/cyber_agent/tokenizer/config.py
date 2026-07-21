"""Strongly typed, hashable tokenizer training configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from cyber_agent.data_pipeline.config import default_project_root
from cyber_agent.data_pipeline.schemas import CATEGORIES, canonical_json


DEFAULT_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|terminal|>",
    "<|document|>",
    "<|code|>",
)


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    project_root: Path
    algorithm: Literal["byte_level_bpe"]
    vocabulary_size: int
    candidate_vocabulary_sizes: tuple[int, ...]
    minimum_frequency: int
    special_tokens: tuple[str, ...]
    seed: int
    maximum_document_characters: int
    category_sampling_weights: dict[str, float]
    category_sampling_limits: dict[str, int]
    source_sampling_limits: dict[str, int]
    training_split_path: Path
    output_directory: Path
    unicode_normalization: Literal["none"]
    maximum_token_length: int

    @classmethod
    def load(cls, project_root: Path | None = None) -> TokenizerConfig:
        root = (project_root or default_project_root()).resolve()
        path = root / "config" / "tokenizer.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load tokenizer configuration {path}: {exc}") from exc
        config = cls(
            project_root=root,
            algorithm=value["algorithm"],
            vocabulary_size=int(value["vocabulary_size"]),
            candidate_vocabulary_sizes=tuple(int(size) for size in value["candidate_vocabulary_sizes"]),
            minimum_frequency=int(value["minimum_frequency"]),
            special_tokens=tuple(value["special_tokens"]),
            seed=int(value["seed"]),
            maximum_document_characters=int(value["maximum_document_characters"]),
            category_sampling_weights={name: float(weight) for name, weight in value["category_sampling_weights"].items()},
            category_sampling_limits={name: int(limit) for name, limit in value["category_sampling_limits"].items()},
            source_sampling_limits={name: int(limit) for name, limit in value["source_sampling_limits"].items()},
            training_split_path=(root / value["training_split_path"]).resolve(),
            output_directory=(root / value["output_directory"]).resolve(),
            unicode_normalization=value["unicode_normalization"],
            maximum_token_length=int(value["maximum_token_length"]),
        )
        config.validate()
        config.candidates_directory.mkdir(parents=True, exist_ok=True)
        config.final_directory.mkdir(parents=True, exist_ok=True)
        return config

    @property
    def candidates_directory(self) -> Path:
        return self.output_directory / "candidates"

    @property
    def final_directory(self) -> Path:
        return self.output_directory / "final"

    @property
    def candidate_directory(self) -> Path:
        return self.candidates_directory / str(self.vocabulary_size)

    def with_overrides(self, *, vocabulary_size: int | None = None, seed: int | None = None) -> TokenizerConfig:
        result = replace(
            self,
            vocabulary_size=self.vocabulary_size if vocabulary_size is None else vocabulary_size,
            seed=self.seed if seed is None else seed,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.algorithm != "byte_level_bpe":
            raise ValueError("only byte_level_bpe is supported")
        if self.vocabulary_size < 300:
            raise ValueError("vocabulary_size must be at least 300 for byte coverage and reserved tokens")
        if self.minimum_frequency < 1:
            raise ValueError("minimum_frequency must be positive")
        if self.maximum_document_characters < 1 or self.maximum_token_length < 1:
            raise ValueError("document and token limits must be positive")
        if self.special_tokens != DEFAULT_SPECIAL_TOKENS:
            raise ValueError("special tokens or their stable order differ from the Phase 3 contract")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")
        if set(self.category_sampling_weights) != CATEGORIES:
            raise ValueError("category_sampling_weights must cover every data category")
        if set(self.category_sampling_limits) != CATEGORIES:
            raise ValueError("category_sampling_limits must cover every data category")
        if any(weight <= 0 for weight in self.category_sampling_weights.values()):
            raise ValueError("category sampling weights must be positive")
        if any(limit < 1 for limit in (*self.category_sampling_limits.values(), *self.source_sampling_limits.values())):
            raise ValueError("sampling limits must be positive")
        expected_train = (self.project_root / "data" / "splits" / "train.jsonl").resolve()
        if self.training_split_path != expected_train:
            raise ValueError("tokenizer training_split_path must be exactly data/splits/train.jsonl")
        if self.unicode_normalization != "none":
            raise ValueError("Phase 3 applies no tokenizer-side Unicode normalization")

    def resolved_dict(self) -> dict[str, Any]:
        def relative(path: Path) -> str:
            try:
                return str(path.relative_to(self.project_root))
            except ValueError:
                return str(path)

        return {
            "algorithm": self.algorithm,
            "vocabulary_size": self.vocabulary_size,
            "candidate_vocabulary_sizes": list(self.candidate_vocabulary_sizes),
            "minimum_frequency": self.minimum_frequency,
            "special_tokens": list(self.special_tokens),
            "seed": self.seed,
            "maximum_document_characters": self.maximum_document_characters,
            "category_sampling_weights": dict(sorted(self.category_sampling_weights.items())),
            "category_sampling_limits": dict(sorted(self.category_sampling_limits.items())),
            "source_sampling_limits": dict(sorted(self.source_sampling_limits.items())),
            "training_split_path": relative(self.training_split_path),
            "output_directory": relative(self.output_directory),
            "unicode_normalization": self.unicode_normalization,
            "maximum_token_length": self.maximum_token_length,
        }

    def configuration_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.resolved_dict()).encode("utf-8")).hexdigest()

