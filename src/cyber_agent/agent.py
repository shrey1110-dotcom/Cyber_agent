"""Model/tool orchestration loop."""

from __future__ import annotations

import logging

from cyber_agent.audit import audit_event
from cyber_agent.model import ModelBackend
from cyber_agent.schema import Message, ModelOutputError, parse_model_action
from cyber_agent.tools import ToolRegistry


class Agent:
    def __init__(
        self,
        backend: ModelBackend,
        tools: ToolRegistry,
        logger: logging.Logger,
        *,
        max_tool_calls: int = 8,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.backend = backend
        self.tools = tools
        self.logger = logger
        self.max_tool_calls = max_tool_calls

    def run(self, user_request: str) -> str:
        audit_event(self.logger, "agent_request", request=user_request)
        if not user_request.strip():
            audit_event(self.logger, "agent_request_rejected", error="empty user request")
            audit_event(self.logger, "agent_completed", status="rejected")
            raise ValueError("user request must not be empty")
        messages = [Message(role="user", content=user_request)]

        tool_calls = 0
        while True:
            try:
                raw = self.backend.generate(messages)
            except Exception as exc:
                audit_event(self.logger, "agent_failure", error=f"model backend failed: {exc}")
                audit_event(self.logger, "agent_completed", status="failure")
                raise
            try:
                action = parse_model_action(raw)
            except ModelOutputError as exc:
                audit_event(self.logger, "model_output_rejected", error=str(exc))
                audit_event(self.logger, "agent_failure", error="invalid model output")
                audit_event(self.logger, "agent_completed", status="failure")
                raise

            messages.append(Message(role="assistant", content=raw))
            if action.type == "final_answer":
                audit_event(self.logger, "agent_completed", status="success")
                return action.content

            if tool_calls >= self.max_tool_calls:
                error = "maximum tool-call count exceeded"
                audit_event(self.logger, "agent_failure", error=error)
                audit_event(self.logger, "agent_completed", status="failure")
                raise RuntimeError(error)
            tool_calls += 1
            tool_result = self.tools.execute(action.tool, action.arguments)
            messages.append(Message(role="tool", content=tool_result.to_json()))
