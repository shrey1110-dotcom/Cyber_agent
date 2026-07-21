from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from cyber_agent.policy import WorkspacePolicy
from cyber_agent.runtime import RuntimeResult
from cyber_agent.tools import ToolRegistry


class RecordingRuntime:
    def __init__(self, result: RuntimeResult | None = None) -> None:
        self.result = result or RuntimeResult(True, "ok", None)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def execute(self, tool: str, arguments: dict[str, Any], timeout: float) -> RuntimeResult:
        self.calls.append((tool, arguments, timeout))
        return self.result


@pytest.fixture
def registry(workspace: Path, audit_logger: logging.Logger) -> tuple[ToolRegistry, RecordingRuntime]:
    runtime = RecordingRuntime()
    return ToolRegistry(WorkspacePolicy(workspace), runtime, audit_logger), runtime


def test_registry_exposes_exactly_five_tools(registry: tuple[ToolRegistry, RecordingRuntime]) -> None:
    tools, _ = registry
    assert set(tools.names) == {
        "list_files",
        "read_file",
        "check_processes",
        "check_ports",
        "run_tests",
    }
    assert len(tools.names) == 5


def test_valid_read_is_normalized_before_runtime(registry: tuple[ToolRegistry, RecordingRuntime]) -> None:
    tools, runtime = registry
    result = tools.execute("read_file", {"path": "README.md"})
    assert result.status == "success"
    assert runtime.calls == [("read_file", {"path": "/workspace/README.md"}, 10.0)]


def test_traversal_is_rejected_without_runtime_call(registry: tuple[ToolRegistry, RecordingRuntime]) -> None:
    tools, runtime = registry
    result = tools.execute("read_file", {"path": "../etc/passwd"})
    assert result.status == "rejected"
    assert "traversal" in (result.error or "")
    assert runtime.calls == []


def test_unknown_tool_is_rejected(registry: tuple[ToolRegistry, RecordingRuntime]) -> None:
    tools, runtime = registry
    result = tools.execute("run_command", {"command": "id"})
    assert result.status == "rejected"
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("check_processes", {"command": "ps aux"}),
        ("check_ports", {"host_network": True}),
        ("run_tests", {"path": ".", "flags": ["--pdb"]}),
        ("read_file", {"path": "README.md", "encoding": "anything"}),
        ("list_files", {"path": ".", "recursive": "yes"}),
    ],
)
def test_rejects_unapproved_arguments(
    registry: tuple[ToolRegistry, RecordingRuntime], tool: str, arguments: dict[str, Any]
) -> None:
    tools, runtime = registry
    assert tools.execute(tool, arguments).status == "rejected"
    assert runtime.calls == []


def test_runtime_failure_is_structured(workspace: Path, audit_logger: logging.Logger) -> None:
    runtime = RecordingRuntime(RuntimeResult(False, "partial", "container failed"))
    tools = ToolRegistry(WorkspacePolicy(workspace), runtime, audit_logger)
    result = tools.execute("check_processes", {})
    assert result.status == "failure"
    assert result.output == "partial"
    assert result.error == "container failed"

