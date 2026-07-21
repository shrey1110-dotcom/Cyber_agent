"""Container entrypoint for the five fixed operations.

This module is not a general command runner. It accepts one JSON request and
dispatches only to the closed operation table below.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from cyber_agent.policy import PolicyRejection, WorkspacePolicy


MAX_OUTPUT_BYTES = 1_048_576
MAX_LIST_ENTRIES = 2_000


def _bounded(text: str) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text
    return encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace") + "\n[output truncated]"


def _resolve_container_path(policy: WorkspacePolicy, value: Any, expected: str) -> Path:
    if not isinstance(value, str):
        raise PolicyRejection("path must be a string")
    return policy.resolve(value, expected=expected)


def list_files(arguments: dict[str, Any], policy: WorkspacePolicy) -> str:
    if set(arguments) != {"path", "recursive"} or not isinstance(arguments["recursive"], bool):
        raise PolicyRejection("invalid list_files arguments")
    root = _resolve_container_path(policy, arguments["path"], "directory")
    iterator = root.rglob("*") if arguments["recursive"] else root.iterdir()
    entries: list[str] = []
    for entry in sorted(iterator, key=lambda item: str(item)):
        if len(entries) >= MAX_LIST_ENTRIES:
            entries.append("[listing truncated]")
            break
        try:
            resolved = entry.resolve(strict=True)
            resolved.relative_to(policy.workspace)
        except (OSError, RuntimeError, ValueError):
            entries.append(f"{entry.relative_to(root)} [blocked symlink]")
            continue
        suffix = "/" if entry.is_dir() else ""
        entries.append(f"{entry.relative_to(root)}{suffix}")
    return "\n".join(entries)


def read_file(arguments: dict[str, Any], policy: WorkspacePolicy) -> str:
    if set(arguments) != {"path"}:
        raise PolicyRejection("invalid read_file arguments")
    path = _resolve_container_path(policy, arguments["path"], "file")
    with path.open("rb") as handle:
        data = handle.read(MAX_OUTPUT_BYTES + 1)
    suffix = b"\n[output truncated]" if len(data) > MAX_OUTPUT_BYTES else b""
    return data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace") + suffix.decode()


def check_processes(arguments: dict[str, Any], policy: WorkspacePolicy) -> str:
    del policy
    if arguments:
        raise PolicyRejection("check_processes accepts no arguments")
    rows = ["pid\tcommand"]
    for item in sorted(Path("/proc").iterdir(), key=lambda path: path.name):
        if not item.name.isdigit():
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
        except (OSError, PermissionError):
            command = "[unavailable]"
        rows.append(f"{item.name}\t{command}")
    return _bounded("\n".join(rows))


def _decode_ipv4(hex_address: str) -> str:
    return socket.inet_ntop(socket.AF_INET, bytes.fromhex(hex_address)[::-1])


def _decode_ipv6(hex_address: str) -> str:
    raw = bytes.fromhex(hex_address)
    words = [raw[index : index + 4][::-1] for index in range(0, 16, 4)]
    return socket.inet_ntop(socket.AF_INET6, b"".join(words))


def check_ports(arguments: dict[str, Any], policy: WorkspacePolicy) -> str:
    del policy
    if arguments:
        raise PolicyRejection("check_ports accepts no arguments")
    rows = ["protocol\tlocal_address\tstate"]
    sources = (("tcp", Path("/proc/net/tcp"), _decode_ipv4), ("tcp6", Path("/proc/net/tcp6"), _decode_ipv6))
    for protocol, path, decoder in sources:
        try:
            lines = path.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            address_hex, port_hex = fields[1].split(":")
            try:
                address = decoder(address_hex)
                port = int(port_hex, 16)
            except (OSError, ValueError):
                continue
            rows.append(f"{protocol}\t{address}:{port}\tLISTEN")
    return "\n".join(rows)


def run_tests(arguments: dict[str, Any], policy: WorkspacePolicy) -> str:
    if set(arguments) != {"path"}:
        raise PolicyRejection("invalid run_tests arguments")
    directory = _resolve_container_path(policy, arguments["path"], "directory")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--disable-warnings", "--maxfail=1", "-q", "."],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=110,
            check=False,
            shell=False,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        raise RuntimeError(f"tests exceeded 110 seconds\n{_bounded(output)}") from exc
    output = _bounded(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"pytest exited with code {completed.returncode}\n{output}")
    return output


OPERATIONS: dict[str, Callable[[dict[str, Any], WorkspacePolicy], str]] = {
    "list_files": list_files,
    "read_file": read_file,
    "check_processes": check_processes,
    "check_ports": check_ports,
    "run_tests": run_tests,
}


def handle_request(raw: str, workspace: Path = Path("/workspace")) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
            raise PolicyRejection("invalid request schema")
        tool = payload["tool"]
        arguments = payload["arguments"]
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise PolicyRejection("invalid request field types")
        operation = OPERATIONS.get(tool)
        if operation is None:
            raise PolicyRejection("unknown tool")
        policy = WorkspacePolicy(workspace)
        output = operation(arguments, policy)
        return {"success": True, "output": output, "error": None}
    except Exception as exc:  # Boundary converts all worker failures to structured data.
        return {"success": False, "output": "", "error": str(exc)}


def main() -> int:
    if len(sys.argv) != 2:
        result = {"success": False, "output": "", "error": "expected exactly one JSON request argument"}
    else:
        result = handle_request(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))
    # Operation failures are valid structured responses for the host runtime.
    # A zero transport exit lets it parse and preserve the worker's error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
