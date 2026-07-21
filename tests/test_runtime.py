from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

from cyber_agent.runtime import DockerRuntime


def test_docker_runtime_applies_hardening_and_never_uses_shell(
    workspace: Path, audit_logger: logging.Logger, monkeypatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"success": True, "output": "safe contents", "error": None}),
            stderr="",
        )

    monkeypatch.setattr("cyber_agent.runtime.subprocess.run", fake_run)
    result = DockerRuntime(workspace, audit_logger).execute(
        "read_file", {"path": "/workspace/README.md"}, 10.0
    )
    command = observed["command"]
    kwargs = observed["kwargs"]
    assert result.success is True
    assert command[:3] == ["docker", "run", "--rm"]
    assert ["--network", "none"] == command[3:5]
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--name") + 1].startswith("cyber-agent-")
    assert command[command.index("--user") + 1] == "10001:10001"
    assert "type=bind" in command[command.index("--mount") + 1]
    assert "readonly" in command[command.index("--mount") + 1]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 10.0


def test_docker_timeout_becomes_failure(
    workspace: Path, audit_logger: logging.Logger, monkeypatch
) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("cyber_agent.runtime.subprocess.run", fake_run)
    result = DockerRuntime(workspace, audit_logger).execute("check_ports", {}, 3.0)
    assert result.success is False
    assert "exceeded 3 seconds" in (result.error or "")
