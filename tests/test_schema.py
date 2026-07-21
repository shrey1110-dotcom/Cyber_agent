from __future__ import annotations

import json

import pytest

from cyber_agent.schema import FinalAnswer, ModelOutputError, ToolCall, ToolResult, parse_model_action


def test_parses_tool_call() -> None:
    raw = json.dumps(
        {
            "type": "tool_call",
            "tool": "read_file",
            "arguments": {"path": "README.md"},
            "reason": "Read documentation.",
        }
    )
    action = parse_model_action(raw)
    assert action == ToolCall("tool_call", "read_file", {"path": "README.md"}, "Read documentation.")


def test_parses_final_answer() -> None:
    assert parse_model_action('{"type":"final_answer","content":"done"}') == FinalAnswer(
        "final_answer", "done"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        '{"type":"tool_call","tool":"read_file","arguments":{},"reason":"x","extra":true}',
        '{"type":"tool_call","tool":"read_file","arguments":[],"reason":"x"}',
        '{"type":"final_answer","content":"x","tool":"read_file"}',
        '{"type":"something_else"}',
    ],
)
def test_rejects_invalid_model_output(raw: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_model_action(raw)


def test_tool_result_wire_format() -> None:
    result = ToolResult("tool_result", "read_file", "success", "contents", None)
    assert json.loads(result.to_json()) == {
        "type": "tool_result",
        "tool": "read_file",
        "status": "success",
        "output": "contents",
        "error": None,
    }

