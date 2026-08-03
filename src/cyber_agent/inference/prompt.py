"""Trusted, bounded prompt construction for the local v0 chat interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.tokenizer.serialization import PromptComponent, TrustedPromptSerializer


DEFAULT_SYSTEM_PROMPT = (
    "You are cyber-agent-llm-v0, a local research assistant. Answer clearly in plain text. "
    "You cannot execute tools, access files, or make network requests."
)

PromptFormat = Literal["plain_v0", "trusted_v1"]


@dataclass(slots=True)
class ChatHistory:
    """Conversation state with an explicit, provenance-matched prompt format."""

    tokenizer: CyberTokenizer
    context_length: int
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_format: PromptFormat = "plain_v0"
    _messages: list[PromptComponent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.context_length < 8:
            raise ValueError("context_length must be at least 8")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if self.prompt_format not in {"plain_v0", "trusted_v1"}:
            raise ValueError("prompt_format must be plain_v0 or trusted_v1")
        if not self._messages:
            self._messages.append(PromptComponent("system", self.system_prompt))

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def reset(self) -> None:
        self._messages = [PromptComponent("system", self.system_prompt)]

    def prompt_for_user(self, user_text: str) -> list[int]:
        """Append a user turn and render a generation prefix without an EOS token.

        User-controlled strings are passed as ordinary content.  Consequently,
        literal strings such as ``<|system|>`` cannot introduce trusted role
        boundaries.
        """
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("chat input must contain non-whitespace text")
        self._messages.append(PromptComponent("user", user_text))
        components = [*self._messages, PromptComponent("assistant", "")]
        if self.prompt_format == "trusted_v1":
            token_ids = TrustedPromptSerializer(self.tokenizer).serialize(
                components,
                add_bos=True,
                add_eos=False,
            )
        else:
            # v0 never trained the role-control embeddings.  Plain labels are
            # ordinary text from the same distribution as its pretraining data.
            labels = {"system": "System", "user": "User", "assistant": "Assistant"}
            rendered = "\n\n".join(
                f"{labels[component.kind]}: {component.canonical_content()}"
                for component in components
            )
            token_ids = self.tokenizer.encode(rendered, add_bos=True, parse_special_tokens=False)
        return self._truncate(token_ids)

    def record_assistant(self, response: str) -> None:
        if not isinstance(response, str):
            raise TypeError("assistant response must be a string")
        self._messages.append(PromptComponent("assistant", response))

    def _truncate(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.context_length:
            return token_ids
        # Preserve an explicit BOS ID and the recent context.  The final
        # assistant boundary is always in the retained suffix.
        return [self.tokenizer.bos_token_id, *token_ids[-(self.context_length - 1):]]
