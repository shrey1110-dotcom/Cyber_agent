"""A decoder-only transformer implemented from random initialization in MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocabulary_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(**value)

    def validate(self) -> None:
        if min(
            self.vocabulary_size, self.context_length, self.hidden_size, self.num_layers,
            self.num_attention_heads, self.intermediate_size,
        ) < 1:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide evenly across attention heads")
        if self.rms_norm_epsilon <= 0:
            raise ValueError("rms_norm_epsilon must be positive")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query = mx.transpose(qkv[:, :, 0], axes=(0, 2, 1, 3))
        key = mx.transpose(qkv[:, :, 1], axes=(0, 2, 1, 3))
        value = mx.transpose(qkv[:, :, 2], axes=(0, 2, 1, 3))
        scores = mx.matmul(query, mx.transpose(key, axes=(0, 1, 3, 2))) * self.scale
        causal_mask = mx.triu(mx.full((sequence_length, sequence_length), -1e9), k=1)
        attention = mx.softmax(scores + causal_mask[None, None, :, :], axis=-1)
        output = mx.matmul(attention, value)
        output = mx.transpose(output, axes=(0, 2, 1, 3)).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.output(output)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(hidden_states)) * self.up(hidden_states))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.mlp = SwiGLU(config)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class CyberDecoderModel(nn.Module):
    """GPT-style causal language model with tied embeddings by default."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.layers = [DecoderBlock(config) for _ in range(config.num_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.output = None if config.tie_word_embeddings else nn.Linear(
            config.hidden_size, config.vocabulary_size, bias=False
        )

    def __call__(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        _, sequence_length = token_ids.shape
        if sequence_length < 1 or sequence_length > self.config.context_length:
            raise ValueError("token sequence length exceeds configured context_length")
        positions = mx.arange(sequence_length)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        if self.output is None:
            return mx.matmul(hidden_states, mx.transpose(self.token_embedding.weight))
        return self.output(hidden_states)

    def parameter_count(self) -> int:
        return sum(int(parameter.size) for _, parameter in tree_flatten(self.parameters()))

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "architecture": "cyber-decoder-v1",
            "causal": True,
            "pre_norm": "RMSNorm",
            "activation": "SwiGLU",
            "learned_position_embeddings": True,
            "tied_input_output_embeddings": self.config.tie_word_embeddings,
            "parameter_count": self.parameter_count(),
            "config": {
                "vocabulary_size": self.config.vocabulary_size,
                "context_length": self.config.context_length,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "intermediate_size": self.config.intermediate_size,
                "rms_norm_epsilon": self.config.rms_norm_epsilon,
                "tie_word_embeddings": self.config.tie_word_embeddings,
            },
        }


def causal_language_model_loss(model: CyberDecoderModel, inputs: mx.array, targets: mx.array) -> mx.array:
    """Mean next-token cross entropy for fixed-length, non-padding blocks."""
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def masked_causal_language_model_loss(
    model: CyberDecoderModel,
    inputs: mx.array,
    targets: mx.array,
    target_weights: mx.array,
) -> mx.array:
    """Next-token loss over explicitly supervised target positions only.

    Instruction prompts are context, not desired completions.  The caller must
    provide a zero/one mask that selects assistant-response tokens and excludes
    prompt and padding positions.
    """
    logits = model(inputs)
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    denominator = mx.maximum(mx.sum(target_weights), mx.array(1.0, dtype=losses.dtype))
    return mx.sum(losses * target_weights) / denominator
