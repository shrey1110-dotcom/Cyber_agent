"""Stable model-independent tokenizer loading, encoding, and decoding API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer


class CyberTokenizer:
    def __init__(
        self,
        backend: Tokenizer,
        special_token_ids: dict[str, int],
        configuration: dict[str, Any],
    ) -> None:
        self._backend = backend
        self._special_token_ids = dict(special_token_ids)
        self.configuration = dict(configuration)

    @classmethod
    def from_file(cls, path: str | Path) -> CyberTokenizer:
        tokenizer_path = Path(path).resolve()
        if not tokenizer_path.exists():
            raise ValueError(f"tokenizer file does not exist: {tokenizer_path}")
        special_path = tokenizer_path.parent / "special_tokens_map.json"
        config_path = tokenizer_path.parent / "tokenizer_config.json"
        if not special_path.exists() or not config_path.exists():
            raise ValueError("tokenizer_config.json and special_tokens_map.json must accompany tokenizer.json")
        try:
            special_payload = json.loads(special_path.read_text(encoding="utf-8"))
            configuration = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load tokenizer metadata: {exc}") from exc
        by_token = special_payload.get("by_token")
        if not isinstance(by_token, dict) or not by_token:
            raise ValueError("special token map is invalid")
        special_ids = {token: int(identifier) for token, identifier in by_token.items()}
        backend = Tokenizer.from_file(str(tokenizer_path))
        for token, identifier in special_ids.items():
            if backend.token_to_id(token) != identifier:
                raise ValueError(f"special token ID mismatch for {token}")
        return cls(backend, special_ids, configuration)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return dict(self._special_token_ids)

    @property
    def vocabulary_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    @property
    def pad_token_id(self) -> int:
        return self._special_token_ids["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self._special_token_ids["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self._special_token_ids["<|eos|>"]

    @property
    def unk_token_id(self) -> int:
        return self._special_token_ids["<|unk|>"]

    def token_id(self, special_token: str) -> int:
        try:
            return self._special_token_ids[special_token]
        except KeyError as exc:
            raise ValueError(f"unknown special token: {special_token}") from exc

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        maximum_length: int | None = None,
        parse_special_tokens: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        content_ids = self._encode_special_safe(text, parse_special_tokens=parse_special_tokens)
        prefix = [self.bos_token_id] if add_bos else []
        suffix = [self.eos_token_id] if add_eos else []
        if maximum_length is not None:
            if maximum_length < 0:
                raise ValueError("maximum_length must not be negative")
            if maximum_length < len(prefix) + len(suffix):
                return (prefix + suffix)[:maximum_length]
            content_ids = content_ids[: maximum_length - len(prefix) - len(suffix)]
        return prefix + content_ids + suffix

    def _encode_special_safe(self, text: str, *, parse_special_tokens: bool) -> list[int]:
        if not text:
            return []
        tokens = tuple(sorted(self._special_token_ids, key=len, reverse=True))
        output: list[int] = []
        cursor = 0
        while cursor < len(text):
            match_position: int | None = None
            match_token: str | None = None
            for token in tokens:
                position = text.find(token, cursor)
                if position != -1 and (match_position is None or position < match_position):
                    match_position = position
                    match_token = token
            if match_position is None or match_token is None:
                output.extend(self._backend.encode(text[cursor:], add_special_tokens=False).ids)
                break
            if match_position > cursor:
                output.extend(self._backend.encode(text[cursor:match_position], add_special_tokens=False).ids)
            if parse_special_tokens:
                output.append(self._special_token_ids[match_token])
            else:
                for character in match_token:
                    output.extend(self._backend.encode(character, add_special_tokens=False).ids)
            cursor = match_position + len(match_token)
        return output

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self._backend.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: Iterable[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        maximum_length: int | None = None,
        padding: bool = False,
        parse_special_tokens: bool = False,
    ) -> dict[str, list[list[int]]]:
        sequences = [
            self.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                maximum_length=maximum_length,
                parse_special_tokens=parse_special_tokens,
            )
            for text in texts
        ]
        target_length = max((len(sequence) for sequence in sequences), default=0) if padding else None
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for sequence in sequences:
            padding_length = 0 if target_length is None else target_length - len(sequence)
            input_ids.append(sequence + [self.pad_token_id] * padding_length)
            attention_masks.append([1] * len(sequence) + [0] * padding_length)
        return {"input_ids": input_ids, "attention_mask": attention_masks}

    def tokens_for_ids(self, token_ids: Iterable[int]) -> list[str]:
        return [self._backend.id_to_token(identifier) or "<invalid-id>" for identifier in token_ids]

