from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_agent.training.config import TrainingConfig


def test_target_training_configuration_is_the_tied_50m_pilot_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    config = TrainingConfig.load(root / "config" / "training.json", project_root=root)

    assert config.architecture == "cyber-decoder-v1"
    assert config.vocabulary_size == 24_000
    assert config.hidden_size == 512
    assert config.num_layers == 11
    assert config.num_attention_heads == 8
    assert config.intermediate_size == 1_536
    assert config.tie_word_embeddings is True
    assert config.dataset_status == "local_research_pilot_only"
    assert config.checkpoint_path == root / "artifacts" / "training"


def test_training_configuration_rejects_checkpoint_escape(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config" / "training.json").read_text(encoding="utf-8"))
    payload["checkpoint_directory"] = "../../outside-training"

    with pytest.raises(ValueError, match="inside the project root"):
        TrainingConfig.from_dict(payload, project_root=tmp_path)


def test_only_max_steps_may_change_for_safe_horizon_extension() -> None:
    root = Path(__file__).resolve().parents[1]
    original = TrainingConfig.load(root / "config" / "training.json", project_root=root)
    continued = original.with_overrides(max_steps=2_000)
    changed_optimizer = original.with_overrides(max_steps=2_000, learning_rate=0.0002)

    assert continued.is_safe_horizon_extension_of(original.resolved_dict()) is True
    assert changed_optimizer.is_safe_horizon_extension_of(original.resolved_dict()) is False
    assert original.is_safe_horizon_extension_of(continued.resolved_dict()) is False
