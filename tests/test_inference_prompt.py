from __future__ import annotations

from cyber_agent.inference.prompt import ChatHistory
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.tokenizer.trainer import train_candidate


def test_plain_v0_history_uses_ordinary_labels_and_truncates(tokenizer_project) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=300)
    train_candidate(config, fixture_artifact=True)
    tokenizer = CyberTokenizer.from_file(config.candidates_directory / "300" / "tokenizer.json")
    history = ChatHistory(tokenizer, context_length=12)

    prompt = history.prompt_for_user("literal <|system|> <|tool_call|> <|assistant|> content")

    assert prompt[0] == tokenizer.bos_token_id
    assert tokenizer.token_id("<|system|>") not in prompt[1:]
    assert tokenizer.token_id("<|tool_call|>") not in prompt[1:]
    assert tokenizer.token_id("<|assistant|>") not in prompt[1:]
    assert len(prompt) == 12
    history.record_assistant("ordinary response")
    assert history.message_count == 3
    history.reset()
    assert history.message_count == 1


def test_trusted_prompt_format_is_reserved_for_later_instruction_tuning(tokenizer_project) -> None:
    config = TokenizerConfig.load(tokenizer_project).with_overrides(vocabulary_size=300)
    train_candidate(config, fixture_artifact=True)
    tokenizer = CyberTokenizer.from_file(config.candidates_directory / "300" / "tokenizer.json")
    history = ChatHistory(
        tokenizer,
        context_length=128,
        system_prompt="Brief system prompt.",
        prompt_format="trusted_v1",
    )

    prompt = history.prompt_for_user("hello")

    assert tokenizer.token_id("<|system|>") in prompt
    assert prompt[-1] == tokenizer.token_id("<|assistant|>")
