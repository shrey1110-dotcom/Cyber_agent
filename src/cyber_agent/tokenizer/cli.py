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
from cyber_agent.tokenizer.trainer import train_candidate


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
    subparsers.add_parser("compare", help="compare all evaluated candidates without auto-selecting")

    export = subparsers.add_parser("export-final", help="explicitly export one evaluated candidate")
    export.add_argument("--candidate", type=int, required=True)

    inspect = subparsers.add_parser("inspect", help="show token IDs and pieces for literal text")
    inspect.add_argument("text")
    inspect.add_argument("--tokenizer", type=Path)
    inspect.add_argument("--parse-special-tokens", action="store_true")
    inspect.add_argument("--bos", action="store_true")
    inspect.add_argument("--eos", action="store_true")
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
        return compare_candidates(config)
    if args.command == "export-final":
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

