from __future__ import annotations

import json
from pathlib import Path

from cyber_agent.sandbox_worker import handle_request


def request(tool: str, arguments: dict[str, object]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


def test_worker_reads_file_with_second_policy_check(workspace: Path) -> None:
    result = handle_request(request("read_file", {"path": "/workspace/README.md"}), workspace)
    assert result == {"success": True, "output": "safe contents\n", "error": None}


def test_worker_rejects_traversal(workspace: Path) -> None:
    result = handle_request(request("read_file", {"path": "../etc/passwd"}), workspace)
    assert result["success"] is False
    assert "traversal" in result["error"]


def test_worker_rejects_unregistered_operation(workspace: Path) -> None:
    result = handle_request(request("shell", {"command": "id"}), workspace)
    assert result["success"] is False
    assert result["error"] == "unknown tool"


def test_worker_lists_files(workspace: Path) -> None:
    result = handle_request(
        request("list_files", {"path": "/workspace", "recursive": False}), workspace
    )
    assert result["success"] is True
    assert result["output"].splitlines() == ["README.md", "src/"]

