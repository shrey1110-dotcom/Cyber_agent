"""Compiled local-only chat runtime for the completed pilot checkpoint.

This is intentionally an inspection interface, not an agent backend.  It never
parses model text as a tool call and has no access to the terminal, Docker, or
network.  The current checkpoint is a private research artifact and may produce
incomplete or low-quality text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_unflatten

from cyber_agent.inference.prompt import ChatHistory, DEFAULT_SYSTEM_PROMPT
from cyber_agent.tokenizer.corpus import sha256_file
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.training.model import CyberDecoderModel, ModelConfig


DEFAULT_V0_RUN_NAME = "pilot-v7-50m-bootstrap-v1"
DEFAULT_V0_STEP = 1000


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside {root}") from exc
    return resolved


def default_checkpoint(project_root: Path) -> Path:
    return (
        project_root / "artifacts" / "training" / DEFAULT_V0_RUN_NAME
        / "checkpoints" / f"step-{DEFAULT_V0_STEP:08d}"
    )


@dataclass(frozen=True, slots=True)
class V0RuntimeInfo:
    checkpoint: Path
    tokenizer_path: Path
    snapshot_name: str
    parameter_count: int
    compiled_forward: bool
    checkpoint_step: int
    local_research_only: bool
    weight_publication_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "cyber-agent-llm-v0",
            "checkpoint": str(self.checkpoint),
            "tokenizer": str(self.tokenizer_path),
            "snapshot_name": self.snapshot_name,
            "parameter_count": self.parameter_count,
            "compiled_forward": self.compiled_forward,
            "checkpoint_step": self.checkpoint_step,
            "local_research_only": self.local_research_only,
            "weight_publication_allowed": self.weight_publication_allowed,
            "tool_execution_enabled": False,
            "network_access_enabled": False,
            "warning": "Pilot-only checkpoint: output is for local inspection, not security advice or autonomous actions.",
        }


class LocalV0ChatModel:
    """Greedy compiled generation over a verified, frozen pilot checkpoint."""

    def __init__(
        self,
        *,
        model: CyberDecoderModel,
        tokenizer: CyberTokenizer,
        info: V0RuntimeInfo,
        compiled: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.history = ChatHistory(tokenizer, context_length=model.config.context_length)
        self._forward = mx.compile(lambda token_ids: self.model(token_ids)) if compiled else self.model

    @classmethod
    def load(
        cls,
        *,
        project_root: Path,
        checkpoint: Path | None = None,
        tokenizer_path: Path | None = None,
        compiled: bool = True,
    ) -> "LocalV0ChatModel":
        root = project_root.resolve()
        training_root = root / "artifacts" / "training"
        checkpoint_dir = _inside(checkpoint or default_checkpoint(root), training_root, label="checkpoint")
        manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        run_manifest_path = checkpoint_dir / "run_manifest.json"
        try:
            checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"v0 checkpoint is incomplete or invalid: {exc}") from exc
        checksums = checkpoint_manifest.get("checksums")
        if not isinstance(checksums, dict) or "model.safetensors" not in checksums:
            raise ValueError("v0 checkpoint lacks required checksum metadata")
        for name, expected_hash in checksums.items():
            path = checkpoint_dir / str(name)
            if not path.is_file() or not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ValueError(f"v0 checkpoint checksum mismatch: {name}")
        if run_manifest.get("run_status") != "local_research_pilot":
            raise ValueError("v0 loader accepts only the explicitly labeled local research pilot checkpoint")
        if run_manifest.get("runtime", {}).get("pretrained_model_used") is not False:
            raise ValueError("v0 checkpoint provenance does not establish random initialization")
        architecture = run_manifest.get("model_architecture", {})
        model_data = architecture.get("config") if isinstance(architecture, dict) else None
        if not isinstance(model_data, dict):
            raise ValueError("v0 checkpoint has no model architecture configuration")
        training_data = run_manifest.get("training_data", {})
        if not isinstance(training_data, dict) or not training_data.get("train_only_verified"):
            raise ValueError("v0 checkpoint provenance does not establish train-only input")
        recorded_tokenizer = training_data.get("tokenizer_path")
        if not isinstance(recorded_tokenizer, str):
            raise ValueError("v0 checkpoint lacks tokenizer provenance")
        selected_tokenizer = _inside(
            tokenizer_path or root / recorded_tokenizer,
            root / "artifacts" / "tokenizers",
            label="tokenizer",
        )
        if sha256_file(selected_tokenizer) != training_data.get("tokenizer_hash"):
            raise ValueError("v0 tokenizer does not match the checkpoint provenance")
        tokenizer = CyberTokenizer.from_file(selected_tokenizer)
        if tokenizer.vocabulary_size != int(model_data.get("vocabulary_size", -1)):
            raise ValueError("v0 tokenizer vocabulary does not match the checkpoint architecture")
        model = CyberDecoderModel(ModelConfig.from_dict(model_data))
        model.update(tree_unflatten(mx.load(checkpoint_dir / "model.safetensors")))
        mx.eval(model.parameters())
        info = V0RuntimeInfo(
            checkpoint=checkpoint_dir,
            tokenizer_path=selected_tokenizer,
            snapshot_name=str(training_data.get("snapshot_name", "unknown")),
            parameter_count=model.parameter_count(),
            compiled_forward=compiled,
            checkpoint_step=int(checkpoint_manifest.get("step", -1)),
            local_research_only=bool(training_data.get("local_research_only")),
            weight_publication_allowed=bool(run_manifest.get("weight_publication_allowed")),
        )
        return cls(model=model, tokenizer=tokenizer, info=info, compiled=compiled)

    def reset(self, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.history = ChatHistory(
            self.tokenizer,
            context_length=self.model.config.context_length,
            system_prompt=system_prompt,
        )

    def reply(self, user_text: str, *, max_new_tokens: int = 64) -> str:
        if max_new_tokens < 1 or max_new_tokens > 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        context = self.history.prompt_for_user(user_text)
        generated: list[int] = []
        special_ids = set(self.tokenizer.special_token_ids.values())
        for _ in range(max_new_tokens):
            next_token = self._next_token(context)
            if next_token == self.tokenizer.eos_token_id or next_token in special_ids:
                break
            generated.append(next_token)
            context = self._trim_context([*context, next_token])
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        self.history.record_assistant(response)
        return response or "[v0 generation stopped without printable text]"

    def _next_token(self, context: list[int]) -> int:
        maximum = self.model.config.context_length
        context = self._trim_context(context)
        padded = [*context, *([self.tokenizer.pad_token_id] * (maximum - len(context)))]
        logits = self._forward(mx.array([padded], dtype=mx.int32))
        next_id = mx.argmax(logits[0, len(context) - 1])
        mx.eval(next_id)
        return int(next_id.item())

    def _trim_context(self, token_ids: list[int]) -> list[int]:
        maximum = self.model.config.context_length
        if len(token_ids) <= maximum:
            return token_ids
        return [self.tokenizer.bos_token_id, *token_ids[-(maximum - 1):]]
