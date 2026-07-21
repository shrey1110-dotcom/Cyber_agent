"""Abstract model boundary and temporary deterministic test backend."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol

from cyber_agent.schema import Message


class ModelBackend(Protocol):
    """Interface that the later local MLX model backend will implement."""

    def generate(self, messages: list[Message]) -> str:
        ...


class TemporaryDeterministicMockBackend:
    """TEMPORARY TEST INFRASTRUCTURE; this is not the final model.

    Responses are returned verbatim in order, making agent-loop tests fully
    deterministic and ensuring that no hosted service is contacted.
    """

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = deque(responses)
        self.received_messages: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.received_messages.append(list(messages))
        if not self._responses:
            raise RuntimeError("temporary mock backend has no response remaining")
        return self._responses.popleft()

