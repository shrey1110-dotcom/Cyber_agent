"""Small, permissively reusable fixture demonstrating safe command arguments."""

from __future__ import annotations

import subprocess


def python_version() -> str:
    completed = subprocess.run(
        ["python", "--version"],
        capture_output=True,
        check=False,
        shell=False,
        text=True,
        timeout=5,
    )
    return (completed.stdout or completed.stderr).strip()

