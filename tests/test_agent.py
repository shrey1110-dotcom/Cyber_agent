from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cyber_agent.agent import Agent
from cyber_agent.model import TemporaryDeterministicMockBackend
from cyber_agent.policy import WorkspacePolicy
from cyber_agent.runtime import RuntimeResult
from cyber_agent.tools import ToolRegistry


class SuccessfulRuntime:
    def execute(self, tool: str, arguments: dict[str, Any], timeout: float) -> RuntimeResult:
        del tool, arguments, timeout
        return RuntimeResult(True, "safe contents\n")


def test_agent_performs_valid_tool_round_trip(workspace: Path, audit_logger: logging.Logger) -> None:
    backend = TemporaryDeterministicMockBackend(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "read_file",
                    "arguments": {"path": "README.md"},
                    "reason": "Read documentation.",
                }
            ),
            json.dumps({"type": "final_answer", "content": "The file is safe."}),
        ]
    )
    registry = ToolRegistry(WorkspacePolicy(workspace), SuccessfulRuntime(), audit_logger)
    answer = Agent(backend, registry, audit_logger).run("Inspect the readme")
    assert answer == "The file is safe."
    tool_message = backend.received_messages[1][-1]
    assert tool_message.role == "tool"
    assert json.loads(tool_message.content) == {
        "type": "tool_result",
        "tool": "read_file",
        "status": "success",
        "output": "safe contents\n",
        "error": None,
    }


def test_agent_returns_rejection_to_model_without_executing(
    workspace: Path, audit_logger: logging.Logger
) -> None:
    backend = TemporaryDeterministicMockBackend(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "read_file",
                    "arguments": {"path": "../etc/passwd"},
                    "reason": "Attempt escape.",
                }
            ),
            json.dumps({"type": "final_answer", "content": "Request rejected."}),
        ]
    )
    registry = ToolRegistry(WorkspacePolicy(workspace), SuccessfulRuntime(), audit_logger)
    assert Agent(backend, registry, audit_logger).run("Escape") == "Request rejected."
    result = json.loads(backend.received_messages[1][-1].content)
    assert result["status"] == "rejected"
    assert "traversal" in result["error"]

