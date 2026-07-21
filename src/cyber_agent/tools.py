"""The closed registry containing exactly five allowed tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from cyber_agent.audit import audit_event
from cyber_agent.policy import PolicyRejection, WorkspacePolicy
from cyber_agent.runtime import ToolRuntime
from cyber_agent.schema import ToolResult


Validator = Callable[[dict[str, Any], WorkspacePolicy], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    timeout_seconds: float
    validate: Validator


def _exact_arguments(arguments: dict[str, Any], allowed: set[str]) -> None:
    extra = set(arguments) - allowed
    if extra:
        raise PolicyRejection(f"unexpected arguments: {', '.join(sorted(extra))}")


def _path_argument(
    arguments: dict[str, Any],
    policy: WorkspacePolicy,
    *,
    expected: str,
    default: str | None = None,
) -> dict[str, Any]:
    _exact_arguments(arguments, {"path"})
    raw = arguments.get("path", default)
    if raw is None:
        raise PolicyRejection("missing required argument: path")
    resolved = policy.resolve(raw, expected=expected)
    return {"path": policy.container_path(resolved)}


def _list_files(arguments: dict[str, Any], policy: WorkspacePolicy) -> dict[str, Any]:
    _exact_arguments(arguments, {"path", "recursive"})
    raw = arguments.get("path", ".")
    recursive = arguments.get("recursive", False)
    if not isinstance(recursive, bool):
        raise PolicyRejection("recursive must be a boolean")
    resolved = policy.resolve(raw, expected="directory")
    return {"path": policy.container_path(resolved), "recursive": recursive}


def _read_file(arguments: dict[str, Any], policy: WorkspacePolicy) -> dict[str, Any]:
    return _path_argument(arguments, policy, expected="file")


def _no_arguments(arguments: dict[str, Any], policy: WorkspacePolicy) -> dict[str, Any]:
    del policy
    _exact_arguments(arguments, set())
    return {}


def _run_tests(arguments: dict[str, Any], policy: WorkspacePolicy) -> dict[str, Any]:
    return _path_argument(arguments, policy, expected="directory", default=".")


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("list_files", 10.0, _list_files),
    ToolSpec("read_file", 10.0, _read_file),
    ToolSpec("check_processes", 10.0, _no_arguments),
    ToolSpec("check_ports", 10.0, _no_arguments),
    ToolSpec("run_tests", 120.0, _run_tests),
)


class ToolRegistry:
    def __init__(
        self,
        policy: WorkspacePolicy,
        runtime: ToolRuntime,
        logger: logging.Logger,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.logger = logger
        self._tools = {spec.name: spec for spec in TOOL_SPECS}
        if len(self._tools) != 5:
            raise RuntimeError("the safe agent shell must expose exactly five tools")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        audit_event(self.logger, "tool_request", tool=tool, arguments=arguments)
        spec = self._tools.get(tool)
        if spec is None:
            error = f"unknown tool: {tool}"
            audit_event(self.logger, "tool_rejected", tool=tool, error=error)
            audit_event(self.logger, "tool_completed", tool=tool, status="rejected")
            return ToolResult("tool_result", tool, "rejected", "", error)

        try:
            validated = spec.validate(arguments, self.policy)
        except (PolicyRejection, TypeError, ValueError) as exc:
            error = str(exc)
            audit_event(self.logger, "tool_rejected", tool=tool, error=error)
            audit_event(self.logger, "tool_completed", tool=tool, status="rejected")
            return ToolResult("tool_result", tool, "rejected", "", error)

        try:
            runtime_result = self.runtime.execute(tool, validated, spec.timeout_seconds)
        except Exception as exc:
            error = f"tool runtime failed unexpectedly: {exc}"
            audit_event(self.logger, "tool_failure", tool=tool, error=error)
            audit_event(self.logger, "tool_completed", tool=tool, status="failure")
            return ToolResult("tool_result", tool, "failure", "", error)
        if not runtime_result.success:
            error = runtime_result.error or "tool execution failed"
            audit_event(self.logger, "tool_failure", tool=tool, error=error)
            audit_event(self.logger, "tool_completed", tool=tool, status="failure")
            return ToolResult("tool_result", tool, "failure", runtime_result.output, error)

        audit_event(self.logger, "tool_completed", tool=tool, status="success")
        return ToolResult("tool_result", tool, "success", runtime_result.output, None)
