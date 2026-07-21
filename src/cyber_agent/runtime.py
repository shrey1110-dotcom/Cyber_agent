"""Docker-only production runtime for fixed sandbox tools."""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cyber_agent.audit import audit_event


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    success: bool
    output: str
    error: str | None = None


class ToolRuntime(Protocol):
    def execute(self, tool: str, arguments: dict[str, Any], timeout: float) -> RuntimeResult:
        ...


class DockerRuntime:
    """Execute a fixed tool request in a fresh, hardened Docker container."""

    def __init__(
        self,
        workspace: Path,
        logger: logging.Logger,
        *,
        image: str = "cyber-agent-sandbox:latest",
        docker_binary: str = "docker",
    ) -> None:
        self.workspace = workspace.resolve(strict=True)
        self.logger = logger
        self.image = image
        self.docker_binary = docker_binary

    def execute(self, tool: str, arguments: dict[str, Any], timeout: float) -> RuntimeResult:
        request_json = json.dumps({"tool": tool, "arguments": arguments}, separators=(",", ":"))
        container_name = f"cyber-agent-{secrets.token_hex(8)}"
        command = [
            self.docker_binary,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            "512m",
            "--cpus",
            "1.0",
            "--name",
            container_name,
            "--user",
            "10001:10001",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--mount",
            f"type=bind,src={self.workspace},dst=/workspace,readonly",
            self.image,
            request_json,
        ]
        audit_event(self.logger, "execution_started", tool=tool, timeout_seconds=timeout)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            audit_event(self.logger, "execution_failed", tool=tool, error="timeout")
            self._force_remove(container_name, tool)
            return RuntimeResult(False, "", f"tool execution exceeded {timeout:g} seconds")
        except OSError as exc:
            audit_event(self.logger, "execution_failed", tool=tool, error=str(exc))
            return RuntimeResult(False, "", f"Docker execution failed: {exc}")

        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"container exited with code {completed.returncode}"
            audit_event(
                self.logger,
                "execution_failed",
                tool=tool,
                container_exit_code=completed.returncode,
            )
            return RuntimeResult(False, completed.stdout.strip(), detail[:4096])

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            audit_event(self.logger, "execution_failed", tool=tool, error="invalid JSON response")
            return RuntimeResult(False, "", "sandbox returned invalid JSON")
        if not isinstance(payload, dict) or set(payload) != {"success", "output", "error"}:
            audit_event(self.logger, "execution_failed", tool=tool, error="invalid result schema")
            return RuntimeResult(False, "", "sandbox returned an invalid result schema")
        if not isinstance(payload["success"], bool):
            audit_event(self.logger, "execution_failed", tool=tool, error="invalid success field")
            return RuntimeResult(False, "", "sandbox result success field is invalid")
        if not isinstance(payload["output"], str):
            audit_event(self.logger, "execution_failed", tool=tool, error="invalid output field")
            return RuntimeResult(False, "", "sandbox result output field is invalid")
        if payload["error"] is not None and not isinstance(payload["error"], str):
            audit_event(self.logger, "execution_failed", tool=tool, error="invalid error field")
            return RuntimeResult(False, "", "sandbox result error field is invalid")
        audit_event(self.logger, "execution_completed", tool=tool, success=payload["success"])
        return RuntimeResult(payload["success"], payload["output"], payload["error"])

    def _force_remove(self, container_name: str, tool: str) -> None:
        """Best-effort cleanup of the exact container created by this call."""
        try:
            cleanup = subprocess.run(
                [self.docker_binary, "rm", "--force", container_name],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            audit_event(self.logger, "execution_cleanup_failed", tool=tool, error=str(exc))
            return
        audit_event(
            self.logger,
            "execution_cleanup_completed",
            tool=tool,
            container_exit_code=cleanup.returncode,
        )
