"""Small CLI for deterministic Phase 1 demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyber_agent.agent import Agent
from cyber_agent.audit import configure_audit_logging
from cyber_agent.model import TemporaryDeterministicMockBackend
from cyber_agent.policy import WorkspacePolicy
from cyber_agent.runtime import DockerRuntime
from cyber_agent.tools import ToolRegistry


def _tool_call(tool: str, arguments: dict[str, object], reason: str) -> str:
    return json.dumps({"type": "tool_call", "tool": tool, "arguments": arguments, "reason": reason})


def _final(content: str) -> str:
    return json.dumps({"type": "final_answer", "content": content})


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe local cybersecurity agent shell")
    parser.add_argument("request", help="user request passed to the deterministic demo backend")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--image", default="cyber-agent-sandbox:latest")
    parser.add_argument("--demo-path", default="README.md")
    parser.add_argument("--log-file")
    parser.add_argument(
        "--traversal-demo",
        action="store_true",
        help="emit a deliberately rejected ../ path request",
    )
    args = parser.parse_args()

    logger = configure_audit_logging(args.log_file)
    policy = WorkspacePolicy(args.workspace.resolve())
    runtime = DockerRuntime(policy.workspace, logger, image=args.image)
    registry = ToolRegistry(policy, runtime, logger)
    if args.traversal_demo:
        first = _tool_call("read_file", {"path": "../etc/passwd"}, "Exercise path policy.")
        final = _final("The traversal request was rejected by policy.")
    else:
        first = _tool_call("read_file", {"path": args.demo_path}, "Read the requested workspace file.")
        final = _final("The safe read_file call completed; its structured result is in the conversation.")

    # Temporary deterministic test infrastructure only. Replace this backend
    # with the future MLXCyberModelBackend without changing Agent or ToolRegistry.
    backend = TemporaryDeterministicMockBackend([first, final])
    answer = Agent(backend, registry, logger).run(args.request)
    print(json.dumps({"type": "final_answer", "content": answer}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

