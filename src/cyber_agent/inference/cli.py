"""Command line access to the explicitly local, pilot-only v0 chat runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cyber_agent.inference.v0 import LocalV0ChatModel, default_checkpoint


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local, pilot-only cyber-agent-llm-v0 chat checkpoint")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--uncompiled", action="store_true", help="disable MLX compilation for diagnostic comparison")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="verify provenance and print v0 runtime status")
    prompt = subparsers.add_parser("prompt", help="generate one local greedy reply")
    prompt.add_argument("text")
    subparsers.add_parser("chat", help="start an interactive local chat; use /quit or /reset")
    return parser


def execute(args: argparse.Namespace) -> int:
    root = args.project_root.resolve()
    model = LocalV0ChatModel.load(
        project_root=root,
        checkpoint=args.checkpoint.resolve() if args.checkpoint else default_checkpoint(root),
        tokenizer_path=args.tokenizer.resolve() if args.tokenizer else None,
        compiled=not args.uncompiled,
    )
    if args.command == "info":
        print(json.dumps(model.info.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "prompt":
        response = model.reply(args.text, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"response": response, "runtime": model.info.to_dict()}, ensure_ascii=False, indent=2))
        return 0
    print("cyber-agent-llm-v0 — local research pilot; no tools, no network. Type /quit to exit.")
    while True:
        try:
            text = input("you> ")
        except EOFError:
            print()
            return 0
        if text.strip() in {"/quit", "/exit"}:
            return 0
        if text.strip() == "/reset":
            model.reset()
            print("conversation reset")
            continue
        try:
            print(f"v0> {model.reply(text, max_new_tokens=args.max_new_tokens)}")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
