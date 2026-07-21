"""Filesystem and argument policy enforcement."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class PolicyRejection(ValueError):
    """A request that is well-formed but disallowed by security policy."""


class WorkspacePolicy:
    """Confines requested paths to one resolved workspace directory."""

    def __init__(self, workspace: Path) -> None:
        if not workspace.is_absolute():
            raise ValueError("workspace path must be absolute")
        if not workspace.exists() or not workspace.is_dir():
            raise ValueError("workspace path must be an existing directory")
        self.workspace = workspace.resolve(strict=True)

    def resolve(
        self,
        raw_path: str,
        *,
        must_exist: bool = True,
        expected: str | None = None,
    ) -> Path:
        if not isinstance(raw_path, str):
            raise PolicyRejection("path must be a string")
        if "\x00" in raw_path:
            raise PolicyRejection("null bytes are not allowed in paths")
        if not raw_path:
            raise PolicyRejection("path must not be empty")
        if "\\" in raw_path:
            raise PolicyRejection("backslashes are not allowed in paths")

        posix_path = PurePosixPath(raw_path)
        if ".." in posix_path.parts:
            raise PolicyRejection("path traversal is not allowed")

        if posix_path.is_absolute():
            try:
                relative = posix_path.relative_to("/workspace")
            except ValueError as exc:
                raise PolicyRejection("absolute path is outside /workspace") from exc
        else:
            relative = posix_path

        candidate = self.workspace.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise PolicyRejection(f"path cannot be resolved: {exc}") from exc

        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise PolicyRejection("path resolves outside /workspace") from exc

        if must_exist and not resolved.exists():
            raise PolicyRejection("path does not exist")
        if expected == "file" and not resolved.is_file():
            raise PolicyRejection("path must identify a regular file")
        if expected == "directory" and not resolved.is_dir():
            raise PolicyRejection("path must identify a directory")
        return resolved

    def container_path(self, resolved: Path) -> str:
        relative = resolved.relative_to(self.workspace)
        return str(PurePosixPath("/workspace").joinpath(*relative.parts))

