"""Local MLX decoder-only model and pretraining interfaces.

This package contains only randomly initialized local training code.  It does
not load a pretrained model or make hosted-model requests.  The MLX-backed
model symbols are loaded lazily so configuration/provenance tooling can run in
non-Metal test or inspection environments.
"""

from typing import TYPE_CHECKING

from cyber_agent.training.config import TrainingConfig

if TYPE_CHECKING:
    from cyber_agent.training.model import CyberDecoderModel, ModelConfig

__all__ = ["CyberDecoderModel", "ModelConfig", "TrainingConfig"]


def __getattr__(name: str):
    if name in {"CyberDecoderModel", "ModelConfig"}:
        from cyber_agent.training.model import CyberDecoderModel, ModelConfig

        return {"CyberDecoderModel": CyberDecoderModel, "ModelConfig": ModelConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
