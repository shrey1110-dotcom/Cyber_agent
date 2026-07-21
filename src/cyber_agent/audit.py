"""Structured security-event logging."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


AUDIT_LOGGER_NAME = "cyber_agent.audit"


def configure_audit_logging(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(AUDIT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def audit_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))

