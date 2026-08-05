"""Resumable, bounded token-ID shard materialization.

Completed shards are never deleted when the disk guard trips.  The operation
stops before writing the next shard and can be resumed safely.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from cyber_agent.tokenizer.loader import CyberTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_snapshot(
    tokenizer_path: str | Path,
    snapshot_directory: str | Path,
    output_directory: str | Path,
    *,
    shard_tokens: int = 10_000_000,
    minimum_free_gib: int = 52,
) -> dict[str, Any]:
    """Encode frozen splits into uint32 shards without duplicating source text."""
    if shard_tokens < 1:
        raise ValueError("shard_tokens must be positive")
    if minimum_free_gib <= 48:
        raise ValueError("minimum_free_gib must be greater than the required 48 GiB floor")
    tokenizer = CyberTokenizer.from_file(tokenizer_path)
    snapshot = Path(snapshot_directory).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "materialization_manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "status": "in_progress", "tokenizer": str(Path(tokenizer_path).resolve()),
        "splits": {}, "minimum_free_gib": minimum_free_gib,
    }
    for split in ("train", "validation", "test"):
        source = snapshot / f"{split}_manifest.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"frozen split is missing: {source}")
        split_dir = output / split
        split_dir.mkdir(exist_ok=True)
        completed = existing.setdefault("splits", {}).setdefault(split, {"shards": [], "documents": 0, "tokens": 0})
        start_document = int(completed.get("documents", 0))
        shard = array.array("I")
        shard_index = len(completed["shards"])
        documents = 0
        tokens = 0
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if documents < start_document:
                    documents += 1
                    continue
                if shutil.disk_usage(output).free < minimum_free_gib * 1024**3:
                    existing["status"] = "paused_disk_guard"
                    manifest_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
                    return existing
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"empty text in frozen {split} manifest")
                ids = tokenizer.encode(text)
                shard.extend(ids)
                documents += 1
                tokens += len(ids)
                if len(shard) >= shard_tokens:
                    final = split_dir / f"shard-{shard_index:06d}.u32"
                    with tempfile.NamedTemporaryFile(dir=split_dir, prefix=f".{final.name}.", delete=False) as temp:
                        temp_path = Path(temp.name)
                        shard.tofile(temp)
                        temp.flush(); os.fsync(temp.fileno())
                    os.replace(temp_path, final)
                    completed["shards"].append({"path": final.name, "tokens": len(shard), "sha256": _sha256(final)})
                    completed["documents"] = documents; completed["tokens"] = int(completed.get("tokens", 0)) + len(shard)
                    manifest_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
                    shard = array.array("I"); shard_index += 1
        if shard:
            final = split_dir / f"shard-{shard_index:06d}.u32"
            with tempfile.NamedTemporaryFile(dir=split_dir, prefix=f".{final.name}.", delete=False) as temp:
                temp_path = Path(temp.name); shard.tofile(temp); temp.flush(); os.fsync(temp.fileno())
            os.replace(temp_path, final)
            completed["shards"].append({"path": final.name, "tokens": len(shard), "sha256": _sha256(final)})
            completed["documents"] = documents; completed["tokens"] = int(completed.get("tokens", 0)) + len(shard)
        existing["status"] = "complete"
        manifest_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return existing
