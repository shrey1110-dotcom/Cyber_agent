from __future__ import annotations

from pathlib import Path

import pytest

from cyber_agent.policy import PolicyRejection, WorkspacePolicy


def test_accepts_relative_and_workspace_absolute_paths(workspace: Path) -> None:
    policy = WorkspacePolicy(workspace)
    assert policy.resolve("README.md", expected="file") == workspace / "README.md"
    assert policy.resolve("/workspace/README.md", expected="file") == workspace / "README.md"


@pytest.mark.parametrize(
    "attack",
    [
        "../etc/passwd",
        "src/../../etc/passwd",
        "/etc/passwd",
        "/workspace/../etc/passwd",
        "bad\x00name",
        r"..\etc\passwd",
        "",
    ],
)
def test_rejects_malicious_paths(workspace: Path, attack: str) -> None:
    with pytest.raises(PolicyRejection):
        WorkspacePolicy(workspace).resolve(attack)


def test_rejects_symlink_escape(workspace: Path, tmp_path: Path) -> None:
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(secret)

    with pytest.raises(PolicyRejection, match="outside"):
        WorkspacePolicy(workspace).resolve("escape", expected="file")


def test_accepts_symlink_that_stays_inside(workspace: Path) -> None:
    (workspace / "readme-link").symlink_to(workspace / "README.md")
    assert WorkspacePolicy(workspace).resolve("readme-link", expected="file") == workspace / "README.md"


def test_requires_expected_file_type(workspace: Path) -> None:
    with pytest.raises(PolicyRejection, match="regular file"):
        WorkspacePolicy(workspace).resolve("src", expected="file")

