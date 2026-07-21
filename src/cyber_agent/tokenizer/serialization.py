"""Canonical trusted prompt serialization with injection-safe content encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cyber_agent.data_pipeline.schemas import canonical_json
from cyber_agent.tokenizer.loader import CyberTokenizer


ComponentKind = Literal[
    "system",
    "user",
    "assistant",
    "tool_call",
    "tool_result",
    "terminal_output",
    "retrieved_document",
    "code_block",
]

CONTROL_TOKEN_FOR_KIND: dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool_call": "<|tool_call|>",
    "tool_result": "<|tool_result|>",
    "terminal_output": "<|terminal|>",
    "retrieved_document": "<|document|>",
    "code_block": "<|code|>",
}


@dataclass(frozen=True, slots=True)
class PromptComponent:
    kind: ComponentKind
    content: str | dict[str, Any] | list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return canonical_json(self.content)


class TrustedPromptSerializer:
    """The sole component authorized to insert trusted control-token IDs."""

    format_name = "cyber-agent-trusted-prompt-v1"

    def __init__(self, tokenizer: CyberTokenizer) -> None:
        self.tokenizer = tokenizer

    def serialize(self, components: list[PromptComponent], *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        if not components:
            raise ValueError("trusted prompt must contain at least one component")
        output = [self.tokenizer.bos_token_id] if add_bos else []
        for component in components:
            control_token = CONTROL_TOKEN_FOR_KIND.get(component.kind)
            if control_token is None:
                raise ValueError(f"unsupported prompt component: {component.kind}")
            output.append(self.tokenizer.token_id(control_token))
            # Crucially, parse_special_tokens remains false. Literal control-token
            # strings in user text, documents, logs, and tool output are bytes.
            output.extend(self.tokenizer.encode(component.canonical_content(), parse_special_tokens=False))
        if add_eos:
            output.append(self.tokenizer.eos_token_id)
        return output

    def serialize_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        components: list[PromptComponent] = []
        for message in messages:
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"unsupported trusted message role: {role}")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("message content must be a string")
            components.append(PromptComponent(role, content))
        return self.serialize(components)

    def inspection(self, components: list[PromptComponent]) -> dict[str, Any]:
        identifiers = self.serialize(components)
        return {
            "format": self.format_name,
            "component_count": len(components),
            "token_ids": identifiers,
            "tokens": self.tokenizer.tokens_for_ids(identifiers),
            "trusted_control_inserter": "TrustedPromptSerializer",
            "untrusted_special_token_parsing": False,
        }
