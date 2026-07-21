"""CLI for validation, training, evaluation, comparison, export, and inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cyber_agent.tokenizer.artifacts import export_final_tokenizer
from cyber_agent.tokenizer.config import TokenizerConfig
from cyber_agent.tokenizer.corpus import validate_training_corpus
from cyber_agent.tokenizer.evaluator import compare_candidates, evaluate_tokenizer
from cyber_agent.tokenizer.loader import CyberTokenizer
from cyber_agent.tokenizer.model_budget import estimate_model_budget
from cyber_agent.tokenizer.pilot import (
    compare_snapshot_candidates,
    evaluate_snapshot_candidate,
    export_snapshot_candidate,
    train_snapshot_candidates,
)
from cyber_agent.tokenizer.serialization import PromptComponent, TrustedPromptSerializer
from cyber_agent.tokenizer.trainer import train_candidate
from cyber_agent.provenance import generate_provenance_attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cyber Agent deterministic tokenizer pipeline")
    parser.add_argument("--project-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-corpus", help="validate the Phase 2 train-only tokenizer corpus")

    train = subparsers.add_parser("train", help="train one local byte-level BPE candidate")
    train.add_argument("--vocab-size", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--fixture", action="store_true", help="label a tiny non-production integration artifact")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a tokenizer candidate")
    evaluate.add_argument("--tokenizer", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="compare legacy candidates or one frozen pilot snapshot")
    compare.add_argument("--snapshot")
    compare.add_argument("--hidden-size", type=int, default=512)
    compare.add_argument("--minimum-evaluation-documents", type=int, default=200)
    compare.add_argument("--minimum-training-tokens", type=int, default=1000000)

    train_pilot = subparsers.add_parser("train-candidates", help="train configured candidates from one frozen snapshot")
    train_pilot.add_argument("--snapshot", required=True)
    train_pilot.add_argument("--vocab-size", type=int, action="append", dest="vocab_sizes")
    train_pilot_alias = subparsers.add_parser("train-pilot-candidates", help="train candidates from a frozen local pilot")
    train_pilot_alias.add_argument("--snapshot", required=True)
    train_pilot_alias.add_argument("--vocab-size", type=int, action="append", dest="vocab_sizes")

    evaluate_pilot = subparsers.add_parser("evaluate-candidate", help="evaluate one frozen-snapshot candidate")
    evaluate_pilot.add_argument("--snapshot", required=True)
    evaluate_pilot.add_argument("--candidate", type=int, required=True)

    compare_pilot = subparsers.add_parser("compare-candidates", help="compare candidates with evidence thresholds")
    compare_pilot.add_argument("--snapshot", required=True)
    compare_pilot.add_argument("--hidden-size", type=int, default=512)
    compare_pilot.add_argument("--minimum-evaluation-documents", type=int, default=1000)
    compare_pilot.add_argument("--minimum-training-tokens", type=int, default=10000000)

    model_budget = subparsers.add_parser("model-budget", help="estimate vocabulary costs for the intended model")
    model_budget.add_argument("--hidden-size", type=int, default=512)
    model_budget.add_argument("--target-parameters", type=int, default=50000000)

    export = subparsers.add_parser("export-final", help="explicitly export one evaluated candidate")
    export.add_argument("--candidate", type=int, required=True)
    export.add_argument("--snapshot")
    export.add_argument("--confirm", action="store_true")
    export.add_argument(
        "--status",
        choices=("pilot_only", "production_candidate", "production_frozen"),
        default="pilot_only",
    )

    inspect = subparsers.add_parser("inspect", help="show token IDs and pieces for literal text")
    inspect.add_argument("text")
    inspect.add_argument("--tokenizer", type=Path)
    inspect.add_argument("--parse-special-tokens", action="store_true")
    inspect.add_argument("--bos", action="store_true")
    inspect.add_argument("--eos", action="store_true")
    serialize = subparsers.add_parser("serialize", help="inspect canonical trusted prompt serialization")
    serialize.add_argument("--tokenizer", type=Path, required=True)
    serialize.add_argument("--kind", choices=(
        "system", "user", "assistant", "tool_call", "tool_result", "terminal_output",
        "retrieved_document", "code_block",
    ), required=True)
    serialize.add_argument("text")
    attest = subparsers.add_parser("attest", help="create local tokenizer provenance metadata without uploading")
    attest.add_argument("--directory", type=Path, required=True)
    attest.add_argument("--signer-identity")
    attest.add_argument("--detached-signature", type=Path)
    return parser


def execute(args: argparse.Namespace) -> Any:
    config = TokenizerConfig.load(args.project_root)
    if args.command == "validate-corpus":
        corpus = validate_training_corpus(config)
        return {
            "status": "valid",
            "input_split": str(config.training_split_path),
            "document_count": corpus.document_count,
            "character_count": corpus.character_count,
            "byte_count": corpus.byte_count,
            "source_counts": corpus.source_counts,
            "category_counts": corpus.category_counts,
            "license_counts": corpus.license_counts,
            "excluded_split_document_counts": corpus.excluded_split_counts,
        }
    if args.command == "train":
        selected = config.with_overrides(vocabulary_size=args.vocab_size, seed=args.seed)
        return train_candidate(selected, fixture_artifact=args.fixture)
    if args.command == "evaluate":
        return evaluate_tokenizer(config, args.tokenizer)
    if args.command == "compare":
        if args.snapshot:
            return compare_snapshot_candidates(
                config,
                snapshot_name=args.snapshot,
                hidden_size=args.hidden_size,
                minimum_evaluation_documents=args.minimum_evaluation_documents,
                minimum_training_estimated_tokens=args.minimum_training_tokens,
            )
        return compare_candidates(config)
    if args.command in {"train-candidates", "train-pilot-candidates"}:
        return train_snapshot_candidates(config, snapshot_name=args.snapshot, vocabulary_sizes=args.vocab_sizes)
    if args.command == "evaluate-candidate":
        return evaluate_snapshot_candidate(config, snapshot_name=args.snapshot, candidate_size=args.candidate)
    if args.command == "compare-candidates":
        return compare_snapshot_candidates(
            config,
            snapshot_name=args.snapshot,
            hidden_size=args.hidden_size,
            minimum_evaluation_documents=args.minimum_evaluation_documents,
            minimum_training_estimated_tokens=args.minimum_training_tokens,
        )
    if args.command == "model-budget":
        return estimate_model_budget(
            config.candidate_vocabulary_sizes,
            hidden_size=args.hidden_size,
            target_model_parameters=args.target_parameters,
        )
    if args.command == "export-final":
        if not args.confirm:
            raise ValueError("final export requires explicit --confirm")
        if args.snapshot:
            directory = export_snapshot_candidate(
                config,
                snapshot_name=args.snapshot,
                candidate_size=args.candidate,
                confirm=args.confirm,
                status=args.status,
            )
        else:
            directory = export_final_tokenizer(config, args.candidate)
        return {"status": "complete", "final_directory": str(directory), "candidate": args.candidate}
    if args.command == "inspect":
        tokenizer_path = args.tokenizer or config.final_directory / "tokenizer.json"
        tokenizer = CyberTokenizer.from_file(tokenizer_path)
        token_ids = tokenizer.encode(
            args.text,
            add_bos=args.bos,
            add_eos=args.eos,
            parse_special_tokens=args.parse_special_tokens,
        )
        return {
            "text": args.text,
            "token_ids": token_ids,
            "tokens": tokenizer.tokens_for_ids(token_ids),
            "decoded": tokenizer.decode(token_ids),
            "special_token_parsing": args.parse_special_tokens,
        }
    if args.command == "serialize":
        tokenizer = CyberTokenizer.from_file(args.tokenizer)
        serializer = TrustedPromptSerializer(tokenizer)
        return serializer.inspection([PromptComponent(args.kind, args.text)])
    if args.command == "attest":
        output = generate_provenance_attestation(
            args.directory.resolve(),
            artifact_type="tokenizer_artifact",
            signer_identity=args.signer_identity,
            detached_signature=args.detached_signature,
            output_path=args.directory.resolve().parent / f"{args.directory.resolve().name}.attestation.json",
        )
        return {"status": "complete", "attestation": str(output), "uploaded": False}
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
