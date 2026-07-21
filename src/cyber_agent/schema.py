"""Strict wire formats shared by model, agent, and tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias


class ModelOutputError(ValueError):
    """Raised when model output does not match the action protocol."""


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    type: Literal["tool_call"]
    tool: str
    arguments: dict[str, Any]
    reason: str


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    type: Literal["final_answer"]
    content: str


ModelAction: TypeAlias = ToolCall | FinalAnswer


@dataclass(frozen=True, slots=True)
class ToolResult:
    type: Literal["tool_result"]
    tool: str
    status: Literal["success", "rejected", "failure"]
    output: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def parse_model_action(raw: str) -> ModelAction:
    """Parse one model response using a closed, strict JSON schema."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"model output is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ModelOutputError("model output must be a JSON object")

    action_type = payload.get("type")
    if action_type == "tool_call":
        expected = {"type", "tool", "arguments", "reason"}
        _require_exact_keys(payload, expected)
        if not isinstance(payload["tool"], str) or not payload["tool"]:
            raise ModelOutputError("tool must be a non-empty string")
        if not isinstance(payload["arguments"], dict):
            raise ModelOutputError("arguments must be a JSON object")
        if not isinstance(payload["reason"], str) or not payload["reason"].strip():
            raise ModelOutputError("reason must be a non-empty string")
        return ToolCall(
            type="tool_call",
            tool=payload["tool"],
            arguments=payload["arguments"],
            reason=payload["reason"],
        )

    if action_type == "final_answer":
        _require_exact_keys(payload, {"type", "content"})
        if not isinstance(payload["content"], str):
            raise ModelOutputError("content must be a string")
        return FinalAnswer(type="final_answer", content=payload["content"])

    raise ModelOutputError("type must be 'tool_call' or 'final_answer'")


def _require_exact_keys(payload: dict[str, Any], expected: set[str]) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ModelOutputError("; ".join(details))

