"""Atomic MLX checkpoint persistence with data/tokenizer provenance checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from cyber_agent.data_pipeline.export import atomic_write_json, atomic_write_text
from cyber_agent.data_pipeline.schemas import utc_now
from cyber_agent.tokenizer.corpus import sha256_file


def _flatten(tree: Any) -> dict[str, mx.array]:
    destination: dict[str, mx.array] = {}
    tree_flatten(tree, destination=destination)
    return destination


def _checksums(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def checkpoint_directory(run_directory: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    return run_directory / "checkpoints" / f"step-{step:08d}"


def save_checkpoint(
    *,
    run_directory: Path,
    step: int,
    model: Any,
    optimizer: Any,
    run_manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> Path:
    """Write a complete checkpoint to a temporary directory then atomically rename."""
    target = checkpoint_directory(run_directory, step)
    if target.exists():
        raise ValueError(f"checkpoint already exists and is immutable: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        assert temporary is not None
        mx.save_safetensors(temporary / "model.safetensors", _flatten(model.parameters()))
        mx.save_safetensors(temporary / "optimizer.safetensors", _flatten(optimizer.state))
        atomic_write_json(temporary / "run_manifest.json", run_manifest)
        atomic_write_text(
            temporary / "metrics.jsonl",
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metrics),
        )
        checksum_names = ("model.safetensors", "optimizer.safetensors", "run_manifest.json", "metrics.jsonl")
        checksums = _checksums(temporary, checksum_names)
        atomic_write_text(
            temporary / "checksums.sha256",
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        )
        manifest = {
            "schema_version": 1,
            "step": step,
            "created_at": utc_now(),
            "checksums": checksums,
            "run_manifest_hash": hashlib.sha256(
                json.dumps(run_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        atomic_write_json(temporary / "checkpoint_manifest.json", manifest)
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint(*, path: str | Path, model: Any, optimizer: Any, expected_run_manifest: dict[str, Any]) -> dict[str, Any]:
    directory = Path(path).resolve()
    manifest_path = directory / "checkpoint_manifest.json"
    try:
        checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint is incomplete or invalid: {exc}") from exc
    for name, expected_hash in checkpoint.get("checksums", {}).items():
        if not isinstance(name, str) or not isinstance(expected_hash, str) or sha256_file(directory / name) != expected_hash:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    for key in ("training_data", "training_configuration_hash", "model_architecture"):
        if run_manifest.get(key) != expected_run_manifest.get(key):
            raise ValueError(f"checkpoint is incompatible with current run: {key}")
    model.update(tree_unflatten(mx.load(directory / "model.safetensors")))
    optimizer.state = tree_unflatten(mx.load(directory / "optimizer.safetensors"))
    mx.eval(model.parameters(), optimizer.state)
    return {"step": int(checkpoint["step"]), "metrics": _load_metrics(directory / "metrics.jsonl")}


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint metrics are invalid: {exc}") from exc
