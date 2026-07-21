from __future__ import annotations

import logging
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

