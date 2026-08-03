"""Deterministic local MLX pretraining with checkpointed provenance."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from cyber_agent.data_pipeline.export import atomic_write_json, atomic_write_text
from cyber_agent.data_pipeline.schemas import utc_now
from cyber_agent.training.checkpoint import load_checkpoint, save_checkpoint
from cyber_agent.training.config import TrainingConfig
from cyber_agent.training.data import TrainingArtifacts
from cyber_agent.training.model import CyberDecoderModel, ModelConfig, causal_language_model_loss


def _learning_rate_schedule(config: TrainingConfig):
    final_rate = config.learning_rate * 0.1
    if config.warmup_steps == 0:
        return optim.linear_schedule(config.learning_rate, final_rate, config.max_steps)
    warmup = optim.linear_schedule(0.0, config.learning_rate, config.warmup_steps)
    decay = optim.linear_schedule(config.learning_rate, final_rate, config.max_steps - config.warmup_steps)
    return optim.join_schedules([warmup, decay], [config.warmup_steps])


def _mean_gradients(accumulated: Any, count: int) -> Any:
    return tree_map(lambda value: value / count, accumulated)


@dataclass(slots=True)
class PretrainingRun:
    config: TrainingConfig
    artifacts: TrainingArtifacts
    run_name: str
    model: CyberDecoderModel
    optimizer: Any
    run_directory: Path
    run_manifest: dict[str, Any]
    metrics: list[dict[str, Any]]
    step: int = 0
    tokens_seen: int = 0

    @classmethod
    def create(
        cls,
        *,
        config: TrainingConfig,
        artifacts: TrainingArtifacts,
        run_name: str,
        resume_checkpoint: Path | None = None,
        allow_finished_checkpoint: bool = False,
        allow_horizon_extension: bool = False,
    ) -> "PretrainingRun":
        if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_name):
            raise ValueError("run_name must use only letters, digits, dot, underscore, and hyphen")
        if artifacts.tokenizer.vocabulary_size != config.vocabulary_size:
            raise ValueError(
                "training vocabulary_size must exactly equal the selected tokenizer vocabulary size "
                f"({config.vocabulary_size} != {artifacts.tokenizer.vocabulary_size})"
            )
        mx.random.seed(config.seed)
        model = CyberDecoderModel(ModelConfig.from_dict(config.model_dict()))
        mx.eval(model.parameters())
        optimizer = optim.AdamW(
            learning_rate=_learning_rate_schedule(config),
            weight_decay=config.weight_decay,
            bias_correction=True,
        )
        run_directory = config.checkpoint_path / run_name
        try:
            run_directory.relative_to(config.project_root)
        except ValueError as exc:
            raise ValueError("run directory escapes the project root") from exc
        model_architecture = model.architecture_manifest()
        run_manifest = {
            "schema_version": 1,
            "run_name": run_name,
            "created_at": utc_now(),
            "runtime": {"framework": "MLX", "hosted_llm_used": False, "pretrained_model_used": False},
            "model_architecture": model_architecture,
            "training_configuration": config.resolved_dict(),
            "training_configuration_hash": config.configuration_hash(),
            "training_data": artifacts.provenance(),
            "run_status": "local_research_pilot" if artifacts.provenance()["local_research_only"] else "production_candidate",
            "weight_publication_allowed": bool(artifacts.provenance()["weight_publication_allowed"]),
            "random_initialization": {"seed": config.seed, "pretrained_weights_loaded": False},
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
        if run_directory.exists() and resume_checkpoint is None:
            raise ValueError("run directory already exists; choose a new run name or resume from an immutable checkpoint")
        if allow_horizon_extension and resume_checkpoint is None:
            raise ValueError("training-horizon extension requires an explicit immutable resume checkpoint")
        if allow_horizon_extension and run_directory.exists():
            raise ValueError("training-horizon extension requires a new run name and empty run directory")
        run_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_directory / "run_manifest.json", run_manifest)
        if run_manifest["run_status"] == "local_research_pilot":
            atomic_write_text(
                run_directory / "LOCAL_RESEARCH_ONLY.txt",
                "LOCAL RESEARCH PILOT ONLY\n\nThese randomly initialized weights used a snapshot that is not cleared for public release.\n",
            )
        run = cls(config, artifacts, run_name, model, optimizer, run_directory, run_manifest, [])
        if resume_checkpoint is not None:
            run.resume(
                resume_checkpoint,
                allow_finished_checkpoint=allow_finished_checkpoint,
                allow_horizon_extension=allow_horizon_extension,
            )
        return run

    def resume(
        self,
        checkpoint: Path,
        *,
        allow_finished_checkpoint: bool = False,
        allow_horizon_extension: bool = False,
    ) -> None:
        loaded = load_checkpoint(
            path=checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            expected_run_manifest=self.run_manifest,
            allow_horizon_extension=allow_horizon_extension,
        )
        self.step = int(loaded["step"])
        self.metrics = list(loaded["metrics"])
        if self.step >= self.config.max_steps and not allow_finished_checkpoint:
            raise ValueError("checkpoint has already reached configured max_steps")
        self.tokens_seen = sum(int(row.get("tokens_seen_increment", 0)) for row in self.metrics)
        if loaded["continued_with_horizon_extension"]:
            self.run_manifest["continuation"] = {
                "kind": "safe_max_steps_extension",
                "parent_checkpoint": str(checkpoint),
                "parent_run_name": loaded["parent_run_name"],
                "parent_training_configuration_hash": loaded["parent_training_configuration_hash"],
            }
            atomic_write_json(self.run_directory / "run_manifest.json", self.run_manifest)

    def _train_batches(self) -> Iterator[tuple[mx.array, mx.array]]:
        while True:
            produced = False
            for batch in self.artifacts.batches(
                "train", sequence_length=self.config.context_length, batch_size=self.config.batch_size
            ):
                produced = True
                yield batch
            if not produced:
                raise ValueError("frozen train split does not produce a full training batch at this context/batch size")

    def _step(self, batches: Iterator[tuple[mx.array, mx.array]]) -> dict[str, float | int]:
        accumulated_grads: Any | None = None
        loss_values: list[mx.array] = []
        for _ in range(self.config.gradient_accumulation_steps):
            inputs, targets = next(batches)
            loss, gradients = nn.value_and_grad(self.model, causal_language_model_loss)(self.model, inputs, targets)
            accumulated_grads = gradients if accumulated_grads is None else tree_map(
                lambda first, second: first + second, accumulated_grads, gradients
            )
            loss_values.append(loss)
        assert accumulated_grads is not None
        gradients = _mean_gradients(accumulated_grads, self.config.gradient_accumulation_steps)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, self.config.max_grad_norm)
        self.optimizer.update(self.model, gradients)
        average_loss = sum(loss_values) / len(loss_values)
        mx.eval(average_loss, gradient_norm, self.model.parameters(), self.optimizer.state)
        token_increment = (
            self.config.batch_size * self.config.context_length * self.config.gradient_accumulation_steps
        )
        return {
            "loss": float(average_loss.item()),
            "gradient_norm": float(gradient_norm.item()),
            "tokens_seen_increment": token_increment,
        }

    def evaluate(self, *, split: str = "validation", maximum_batches: int | None = None) -> dict[str, float | int]:
        losses: list[float] = []
        correct = 0
        token_count = 0
        for batch_number, (inputs, targets) in enumerate(
            self.artifacts.batches(split, sequence_length=self.config.context_length, batch_size=self.config.batch_size),
            start=1,
        ):
            logits = self.model(inputs)
            loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
            predictions = mx.argmax(logits, axis=-1)
            matches = mx.sum(predictions == targets)
            mx.eval(loss, matches)
            losses.append(float(loss.item()))
            correct += int(matches.item())
            token_count += int(targets.size)
            if maximum_batches is not None and batch_number >= maximum_batches:
                break
        if not losses:
            return {"split": split, "batches": 0, "loss": None, "perplexity": None, "token_accuracy": None}
        mean_loss = sum(losses) / len(losses)
        return {
            "split": split,
            "batches": len(losses),
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)),
            "token_accuracy": correct / token_count,
        }

    def _save(self) -> Path:
        checkpoint = save_checkpoint(
            run_directory=self.run_directory,
            step=self.step,
            model=self.model,
            optimizer=self.optimizer,
            run_manifest=self.run_manifest,
            metrics=self.metrics,
        )
        atomic_write_json(
            self.run_directory / "latest.json",
            {"step": self.step, "checkpoint": str(checkpoint.relative_to(self.run_directory)), "updated_at": utc_now()},
        )
        return checkpoint

    def train(
        self,
        *,
        max_steps: int | None = None,
        maximum_evaluation_batches: int | None = None,
    ) -> dict[str, Any]:
        requested_steps = self.config.max_steps if max_steps is None else min(max_steps, self.config.max_steps)
        if requested_steps <= self.step:
            raise ValueError("requested training steps must exceed the resumed step")
        batches = self._train_batches()
        last_checkpoint: Path | None = None
        while self.step < requested_steps:
            metric = self._step(batches)
            self.step += 1
            self.tokens_seen += int(metric["tokens_seen_increment"])
            metric.update({"step": self.step, "tokens_seen": self.tokens_seen, "timestamp": utc_now()})
            if self.step % self.config.evaluation_every_steps == 0 or self.step == requested_steps:
                metric["validation"] = self.evaluate(
                    split="validation", maximum_batches=maximum_evaluation_batches
                )
            self.metrics.append(metric)
            if self.step % self.config.checkpoint_every_steps == 0 or self.step == requested_steps:
                last_checkpoint = self._save()
        report = {
            "status": "complete",
            "run_name": self.run_name,
            "steps_completed": self.step,
            "tokens_seen": self.tokens_seen,
            "last_metric": self.metrics[-1],
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            "run_status": self.run_manifest["run_status"],
            "weight_publication_allowed": self.run_manifest["weight_publication_allowed"],
        }
        atomic_write_json(self.run_directory / "training_report.json", report)
        return report
