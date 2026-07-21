from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def audit_logger() -> logging.Logger:
    logger = logging.getLogger("cyber_agent.tests.audit")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("safe contents\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


@pytest.fixture
def pipeline_project(tmp_path: Path) -> Path:
    project = tmp_path / "pipeline-project"
    project.mkdir()
    repository_root = Path(__file__).resolve().parents[1]
    shutil.copytree(repository_root / "config", project / "config")
    shutil.copytree(repository_root / "fixtures", project / "fixtures")
    for name in ("raw", "extracted", "cleaned", "rejected", "manifests", "splits", "reports"):
        (project / "data" / name).mkdir(parents=True)
    return project

