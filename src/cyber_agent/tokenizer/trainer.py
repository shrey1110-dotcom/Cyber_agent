"""Deterministic local byte-level BPE training from the validated train split."""

from __future__ import annotations

from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cyber_agent.tokenizer.artifacts import write_candidate_artifacts
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.corpus import CorpusPlan, validate_training_corpus


def train_candidate(
    config: TokenizerConfig,
    *,
    fixture_artifact: bool = False,
    corpus: CorpusPlan | None = None,
) -> dict[str, Any]:
    plan = corpus or validate_training_corpus(config)
    if plan.document_count < 100 and not fixture_artifact:
        raise ValueError(
            "corpus is too small for a production candidate; use an explicitly labeled fixture artifact for pipeline testing"
        )
    if plan.document_count < 100 and config.vocabulary_size > 2048:
        raise ValueError("fixture corpus is too small for this vocabulary; use a deliberately small size such as 512")

    backend = Tokenizer(
        models.BPE(
            unk_token="<|unk|>",
            byte_fallback=True,
        )
    )
    backend.normalizer = None
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocabulary_size,
        min_frequency=config.minimum_frequency,
        show_progress=False,
        special_tokens=list(config.special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=config.maximum_token_length,
    )
    backend.train_from_iterator(plan.iter_texts(), trainer=trainer, length=plan.document_count)
    special_ids = {token: backend.token_to_id(token) for token in config.special_tokens}
    expected_ids = {token: index for index, token in enumerate(config.special_tokens)}
    if special_ids != expected_ids:
        raise ValueError(f"special token IDs are unstable: expected {expected_ids}, received {special_ids}")

    directory, manifest = write_candidate_artifacts(
        backend,
        config,
        plan,
        fixture_artifact=fixture_artifact,
    )
    return {
        "status": "complete",
        "candidate_directory": str(directory),
        "requested_vocabulary_size": config.vocabulary_size,
        "actual_vocabulary_size": manifest["actual_vocabulary_size"],
        "document_count": plan.document_count,
        "fixture_artifact": fixture_artifact,
        "special_tokens": manifest["special_tokens"],
        "artifact_hashes": manifest["tokenizer_artifact_hashes"],
    }

