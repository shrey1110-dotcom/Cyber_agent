"""Deterministic, non-operational examples for the closed five-tool protocol."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.export import atomic_write_jsonl
from cyber_agent.data_pipeline.schemas import canonical_json
from cyber_agent.data_pipeline.sources import SourceDefinition


GENERATOR_VERSION = "safe-tool-examples-v3"
SAFE_TOOLS: dict[str, tuple[dict[str, Any], str]] = {
    "list_files": ({"path": ".", "recursive": False}, "List files inside the approved workspace."),
    "read_file": ({"path": "README.md"}, "Read one approved workspace document."),
    "check_processes": ({}, "Inspect processes inside the isolated sandbox."),
    "check_ports": ({}, "Inspect locally listening services inside the sandbox."),
    "run_tests": ({"path": "."}, "Run the fixed test command inside the sandbox."),
}
EXAMPLE_KINDS = (
    "valid_tool_call",
    "valid_tool_result",
    "malformed_request",
    "unsupported_tool_refusal",
    "path_traversal_refusal",
    "safe_explanation",
    "confirmation_required_future_capability",
)


def _example(tool: str, kind: str, variant: int) -> dict[str, Any]:
    arguments, reason = SAFE_TOOLS[tool]
    if kind == "valid_tool_call":
        payload: Any = {"type": "tool_call", "tool": tool, "arguments": arguments, "reason": reason}
        explanation = "The request names one registered tool and uses its closed argument schema."
    elif kind == "valid_tool_result":
        payload = {
            "type": "tool_result", "tool": tool, "status": "success",
            "output": f"Safe synthetic result variant {variant}; no host data is included.", "error": None,
        }
        explanation = "A tool result is structured data returned only after sandboxed execution."
    elif kind == "malformed_request":
        payload = {"type": "tool_call", "tool": tool, "arguments": "not-an-object"}
        explanation = "Refuse this request because arguments must be an object and reason is required."
    elif kind == "unsupported_tool_refusal":
        payload = {"type": "tool_call", "tool": "run_shell", "arguments": {"command": "not executed"}}
        explanation = "Refuse because run_shell is not one of the five registered tools; do not execute anything."
    elif kind == "path_traversal_refusal":
        payload = {"type": "tool_call", "tool": "read_file", "arguments": {"path": "../../outside.txt"}}
        explanation = "Refuse because traversal components would escape the approved workspace boundary."
    elif kind == "safe_explanation":
        payload = {"question": f"Why is {tool} bounded?", "answer": reason}
        explanation = "Explain the boundary without inventing commands or expanding permissions."
    elif kind == "confirmation_required_future_capability":
        payload = {"requested_capability": "write_file", "status": "confirmation_required", "executed": False}
        explanation = "A future write capability would require explicit user confirmation and a narrow path policy."
    else:  # pragma: no cover - guarded by the fixed kind tuple
        raise ValueError(f"unsupported synthetic example kind: {kind}")
    rng = random.Random(variant)
    subjects = ("workspace", "documentation", "test suite", "sandbox", "policy boundary", "audit trail", "project tree")
    goals = ("inspection", "verification", "explanation", "troubleshooting", "review", "validation", "teaching")
    constraints = (
        "remain inside the approved workspace", "use only the closed tool schema",
        "avoid credentials and personal data", "never execute arbitrary shell text",
        "preserve a structured audit trail", "refuse traversal outside the sandbox",
        "request confirmation before future write capabilities",
    )
    variations = []
    for index in range(12):
        subject = rng.choice(subjects)
        goal = rng.choice(goals)
        constraint = rng.choice(constraints)
        variations.append(
            f"Practice case {index + 1}: explain {goal} for the {subject}; {constraint}. "
            f"The safe response should classify this as {kind} for {tool} and state why no broader capability is granted."
        )
    nouns = (
        "workspace", "policy", "sandbox", "audit", "schema", "request", "result", "process",
        "port", "document", "directory", "test", "permission", "boundary", "validation", "parser",
        "registry", "timeout", "resource", "confirmation",
    )
    modifiers = (
        "local", "bounded", "structured", "approved", "deterministic", "readable", "isolated", "safe",
        "explicit", "validated", "audited", "restricted", "temporary", "canonical", "portable", "private",
        "atomic", "resumable", "defensive", "documented",
    )
    concepts = [f"{modifier}_{noun}" for modifier in modifiers for noun in nouns]
    rng.shuffle(concepts)
    return {
        "instruction": "Classify and safely handle the structured agent protocol example.",
        "input": payload,
        "response": explanation,
        "safety": {
            "allowed_tools": sorted(SAFE_TOOLS),
            "arbitrary_shell_allowed": False,
            "downloaded_code_executed": False,
        },
        "practice_variations": variations,
        "technical_concepts": concepts[:180],
    }


def generate_safe_tool_examples(
    source: SourceDefinition,
    destination_manifest: Path,
    *,
    seed: int,
    document_count: int = len(SAFE_TOOLS) * len(EXAMPLE_KINDS),
) -> dict[str, Any]:
    """Generate a deterministic local source tree and manifest atomically."""
    if source.adapter != "synthetic_tool_examples":
        raise ValueError("source does not use the synthetic tool-example adapter")
    if document_count < len(SAFE_TOOLS) * len(EXAMPLE_KINDS):
        raise ValueError("synthetic document count is too small to cover every tool and example kind")
    root = destination_manifest.parent
    expected_identity = hashlib.sha256(
        canonical_json({"version": GENERATOR_VERSION, "seed": seed, "documents": document_count}).encode("utf-8")
    ).hexdigest()
    identity_path = root / "generator_manifest.json"
    if destination_manifest.exists() and identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != expected_identity:
            raise ValueError("synthetic output exists with a different generator version, seed, or count")
        return {"status": "skipped", "documents": document_count, "manifest": str(destination_manifest)}
    if root.exists():
        raise ValueError("synthetic destination exists but is incomplete; move it aside before regenerating")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{root.name}.", suffix=".tmp", dir=root.parent))
    try:
        assert temporary is not None
        documents_dir = temporary / "documents"
        documents_dir.mkdir()
        combinations = [(tool, kind) for tool in sorted(SAFE_TOOLS) for kind in EXAMPLE_KINDS]
        rng = random.Random(seed)
        rows: list[dict[str, Any]] = []
        for index in range(document_count):
            tool, kind = combinations[index % len(combinations)]
            variant = rng.randrange(1_000_000)
            record = _example(tool, kind, variant)
            record["provenance"] = {
                "synthetic": True,
                "generator_version": GENERATOR_VERSION,
                "seed": seed,
                "index": index,
                "example_kind": kind,
            }
            relative = Path("documents") / f"example-{index:05d}.json"
            (temporary / relative).write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append({
                "path": relative.as_posix(),
                "source_url": f"local://{source.source_name}/{index}",
                "source_release": source.exact_release_or_version,
                "license": source.license,
                "category": "terminal",
                "language": "en",
                "retrieved_at": source.retrieved_at,
                "media_type": "application/json",
                "metadata": {
                    "synthetic": True,
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed,
                    "example_kind": kind,
                    "tool": tool,
                    "programming_language": "json",
                },
            })
        atomic_write_jsonl(temporary / "manifest.jsonl", rows)
        (temporary / "generator_manifest.json").write_text(
            json.dumps({
                "generator_version": GENERATOR_VERSION,
                "seed": seed,
                "documents": document_count,
                "identity_sha256": expected_identity,
                "allowed_tools": sorted(SAFE_TOOLS),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        temporary = None
        return {"status": "complete", "documents": document_count, "manifest": str(destination_manifest)}
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
