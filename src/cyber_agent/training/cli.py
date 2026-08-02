"""Command-line entry points for explicitly confirmed local MLX pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cyber_agent.training.config import TrainingConfig
from cyber_agent.training.data import TrainingArtifacts
from cyber_agent.training.trainer import PretrainingRun


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config(path: Path | None, root: Path) -> TrainingConfig:
    return TrainingConfig.load(path or root / "config" / "training.json", project_root=root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the random-initialized Cyber Agent decoder locally with MLX")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--config", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-model", help="show target architecture and parameter count")
    inspect.add_argument("--tokenizer", type=Path, required=True)

    train = subparsers.add_parser("train", help="start or resume an explicitly confirmed local training run")
    train.add_argument("--snapshot", required=True)
    train.add_argument("--tokenizer", type=Path, required=True)
    train.add_argument("--run-name", required=True)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--evaluation-max-batches", type=int)
    train.add_argument("--resume", type=Path)
    train.add_argument("--allow-pilot-artifacts", action="store_true")
    train.add_argument("--confirm-start", action="store_true")

    smoke = subparsers.add_parser("smoke-train", help="run a tiny, non-production local mechanics check")
    smoke.add_argument("--snapshot", required=True)
    smoke.add_argument("--tokenizer", type=Path, required=True)
    smoke.add_argument("--run-name", required=True)
    smoke.add_argument("--steps", type=int, default=2)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a held-out frozen split")
    evaluate.add_argument("--snapshot", required=True)
    evaluate.add_argument("--tokenizer", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--allow-pilot-artifacts", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    config = _config(args.config, root)
    if args.command == "inspect-model":
        from cyber_agent.tokenizer.artifacts import verify_candidate
        from cyber_agent.tokenizer.loader import CyberTokenizer
        from cyber_agent.training.model import CyberDecoderModel, ModelConfig
        import mlx.core as mx

        tokenizer_path = args.tokenizer.resolve()
        candidate = verify_candidate(tokenizer_path.parent)
        tokenizer = CyberTokenizer.from_file(tokenizer_path)
        if tokenizer.vocabulary_size != config.vocabulary_size:
            raise ValueError("training configuration vocabulary_size does not match tokenizer")
        mx.random.seed(config.seed)
        model = CyberDecoderModel(ModelConfig.from_dict(config.model_dict()))
        mx.eval(model.parameters())
        return {
            "status": "inspection_only",
            "model": model.architecture_manifest(),
            "tokenizer": {
                "path": str(tokenizer_path), "vocabulary_size": tokenizer.vocabulary_size,
                "snapshot_name": candidate.get("snapshot_name"), "production_ready": candidate.get("production_ready"),
            },
            "no_weights_saved": True,
            "no_training_started": True,
        }

    allow_pilot = bool(getattr(args, "allow_pilot_artifacts", False))
    if args.command == "smoke-train":
        if args.steps < 1 or args.steps > 10:
            raise ValueError("smoke-train --steps must be between 1 and 10")
        artifacts = TrainingArtifacts.load(
            project_root=root, snapshot_name=args.snapshot, tokenizer_path=args.tokenizer, allow_pilot_artifacts=True
        )
        smoke_config = config.smoke_config(vocabulary_size=artifacts.tokenizer.vocabulary_size, max_steps=args.steps)
        run = PretrainingRun.create(config=smoke_config, artifacts=artifacts, run_name=args.run_name)
        return {**run.train(), "smoke_run": True, "production_model": False}

    if args.command == "train":
        if not args.confirm_start:
            raise ValueError("training has not started; pass --confirm-start after reviewing the snapshot and tokenizer provenance")
        artifacts = TrainingArtifacts.load(
            project_root=root, snapshot_name=args.snapshot, tokenizer_path=args.tokenizer, allow_pilot_artifacts=allow_pilot
        )
        run = PretrainingRun.create(
            config=config, artifacts=artifacts, run_name=args.run_name,
            resume_checkpoint=args.resume.resolve() if args.resume else None,
        )
        if args.evaluation_max_batches is not None and args.evaluation_max_batches < 1:
            raise ValueError("--evaluation-max-batches must be positive")
        return run.train(
            max_steps=args.max_steps,
            maximum_evaluation_batches=args.evaluation_max_batches,
        )

    if args.command == "evaluate":
        try:
            checkpoint_run_manifest = json.loads(
                (args.checkpoint.resolve() / "run_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"checkpoint run manifest is unavailable: {exc}") from exc
        stored_config = checkpoint_run_manifest.get("training_configuration")
        if not isinstance(stored_config, dict):
            raise ValueError("checkpoint has no resolved training configuration")
        config = TrainingConfig.from_dict(stored_config, project_root=root)
        artifacts = TrainingArtifacts.load(
            project_root=root, snapshot_name=args.snapshot, tokenizer_path=args.tokenizer, allow_pilot_artifacts=allow_pilot
        )
        run = PretrainingRun.create(
            config=config, artifacts=artifacts, run_name=args.run_name, resume_checkpoint=args.checkpoint.resolve(),
            allow_finished_checkpoint=True,
        )
        return {"status": "complete", "evaluation": run.evaluate(split=args.split), "run_status": run.run_manifest["run_status"]}
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except KeyboardInterrupt:
        print(json.dumps({"status": "failure", "error": "interrupted"}), file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
