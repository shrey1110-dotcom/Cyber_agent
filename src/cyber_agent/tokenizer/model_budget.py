"""Architectural estimates for tokenizer vocabulary parameter cost."""

from __future__ import annotations

from typing import Any, Iterable


def estimate_model_budget(
    vocabulary_sizes: Iterable[int] = (16000, 24000, 32000),
    *,
    hidden_size: int = 512,
    target_model_parameters: int = 50_000_000,
) -> dict[str, Any]:
    if hidden_size < 1 or target_model_parameters < 1:
        raise ValueError("hidden size and target parameter budget must be positive")
    rows: list[dict[str, Any]] = []
    for vocabulary_size in vocabulary_sizes:
        if vocabulary_size < 1:
            raise ValueError("vocabulary sizes must be positive")
        embeddings = vocabulary_size * hidden_size
        for tied in (True, False):
            output_head = 0 if tied else vocabulary_size * hidden_size
            vocabulary_related = embeddings + output_head
            rows.append(
                {
                    "vocabulary_size": vocabulary_size,
                    "hidden_size": hidden_size,
                    "tied_input_output_embeddings": tied,
                    "embedding_parameters": embeddings,
                    "output_head_parameters": output_head,
                    "total_vocabulary_related_parameters": vocabulary_related,
                    "remaining_transformer_parameter_budget": target_model_parameters - vocabulary_related,
                }
            )
    return {
        "schema_version": 1,
        "status": "architectural_estimate",
        "target_model_parameters": target_model_parameters,
        "hidden_size": hidden_size,
        "assumptions": [
            "Embedding cost is vocabulary_size × hidden_size.",
            "A tied output head reuses embedding weights and adds no separate weight matrix.",
            "An untied output head adds vocabulary_size × hidden_size weights.",
            "Biases, optimizer state, activation memory, and final transformer configuration are excluded.",
        ],
        "candidates": rows,
    }
